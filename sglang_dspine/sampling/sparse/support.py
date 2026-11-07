"""DSpine inference kernels and graph-owned buffers."""

import math
from fractions import Fraction

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


def upper(x):
    y = float(x)
    return math.nextafter(y, math.inf) if Fraction.from_float(y) < x else y


def constants(n):
    u = Fraction(1, 2**24)
    a = Fraction(4, 2**126)
    d = 1 - (n - 1) * u
    g = lambda k: k * u / (1 - k * u)
    u64 = Fraction(1, 2**53)
    return dict(
        HIST_SCALE=upper(u / d),
        HIST_ABS=upper(2 * n * a / d),
        TOTAL_GAMMA=upper(g(64)),
        SCAN_GAMMA=upper(g(2176)),
        TOTAL_ABS=upper(4096 * a / (1 - 64 * u)),
        SCAN_ABS=upper(4096 * a / (1 - 2176 * u)),
        U=upper(u),
        A=upper(a),
        GM=upper((n - 1) * u64 / (1 - (n - 1) * u64)),
        GP=upper(19 * u64 / (1 - 19 * u64)),
    )


@triton.jit
def _up(x):
    return libdevice.nextafter(x, tl.full((), float("inf"), tl.float64))


@triton.jit
def _down(x):
    return libdevice.nextafter(x, tl.full((), -float("inf"), tl.float64))


@triton.jit
def _sum_parts(
    S, Mass, Bad, V: tl.constexpr, CHUNKS: tl.constexpr, BLOCK: tl.constexpr
):
    (row, part) = (tl.program_id(0), tl.program_id(1))
    off = part * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(S + row * V + off, off < V, 0.0)
    good = (x >= 0.0) & (tl.abs(x) < float("inf"))
    tl.store(Mass + row * CHUNKS + part, tl.sum(x.to(tl.float64), 0))
    tl.store(Bad + row * CHUNKS + part, tl.sum((~good).to(tl.int32), 0))


@triton.jit
def _certificate(
    Top,
    Mass,
    Bad,
    Ks,
    Ps,
    Enabled,
    Guard,
    Cut,
    Error,
    Rank,
    Onehot,
    V: tl.constexpr,
    CHUNKS: tl.constexpr,
    MASS_BLOCK: tl.constexpr,
    HIST_SCALE: tl.constexpr,
    HIST_ABS: tl.constexpr,
    TOTAL_GAMMA: tl.constexpr,
    SCAN_GAMMA: tl.constexpr,
    TOTAL_ABS: tl.constexpr,
    SCAN_ABS: tl.constexpr,
    U: tl.constexpr,
    A: tl.constexpr,
    GM: tl.constexpr,
    GP: tl.constexpr,
):
    row = tl.program_id(0)
    i = tl.arange(0, 32)
    t = tl.load(Top + row * 32 + i)
    nxt = tl.load(Top + row * 32 + i + 1, i < 31, -float("inf"))
    j = tl.arange(0, MASS_BLOCK)
    mh = tl.sum(tl.load(Mass + row * CHUNKS + j, j < CHUNKS, 0.0), 0)
    bad = tl.sum(tl.load(Bad + row * CHUNKS + j, j < CHUNKS, 0), 0)
    one = tl.full((), 1.0, tl.float64)
    (gm, gp) = (tl.full((), GM, tl.float64), tl.full((), GP, tl.float64))
    mlo = _down(mh / _up(one + gm))
    mhi = _up(mh / _down(one - gm))
    ch = tl.cumsum(tl.where(i < 20, t, 0.0).to(tl.float64), 0)
    clo = _down(ch / _up(one + gp))
    chi = _up(ch / _down(one - gp))
    hhi = tl.sum(tl.where(i == 19, chi, 0.0), 0)
    hlo = tl.sum(tl.where(i == 19, clo, 0.0), 0)
    lhi = tl.maximum(_up(mhi - hlo), 0.0)
    (hs, ha) = (tl.full((), HIST_SCALE, tl.float64), tl.full((), HIST_ABS, tl.float64))
    (tg, sg) = (
        tl.full((), TOTAL_GAMMA, tl.float64),
        tl.full((), SCAN_GAMMA, tl.float64),
    )
    (ta, sa) = (tl.full((), TOTAL_ABS, tl.float64), tl.full((), SCAN_ABS, tl.float64))
    (u, a) = (tl.full((), U, tl.float64), tl.full((), A, tl.float64))
    nineteen = tl.full((), 19.0, tl.float64)
    nv = tl.full((), V + 24.0, tl.float64)
    eh = _up(_up(hs * _up(_up(nineteen * hhi) + _up(nv * lhi))) + ha)
    mt = _up(mhi + eh)
    et = _up(_up(eh + _up(tg * mt)) + ta)
    es = _up(_up(eh + _up(sg * mt)) + sa)
    p32 = tl.load(Ps + row)
    p = p32.to(tl.float64)
    d = _up(_up(_up(p * et) + _up(u * _up(p * _up(mhi + et)))) + a)
    for _ in tl.static_range(2):
        d = _up(_up(_up(_up(one + u) * _up(d + es)) + _up(u * mhi)) + a)
    error = _up(d + es)
    (tlo, thi) = (_down(p * mlo), _up(p * mhi))
    prev_hi = tl.where(i > 0, tl.gather(chi, tl.maximum(i - 1, 0), 0), 0.0)
    fits = (
        (i < 20) & (_up(prev_hi + error) < tlo) & (_down(clo - error) > thi) & (t > nxt)
    )
    index = tl.min(tl.where(fits, i, 32), 0)
    t19 = tl.sum(tl.where(i == 19, t, 0.0), 0)
    t20 = tl.sum(tl.where(i == 20, t, 0.0), 0)
    k = tl.load(Ks + row)
    enabled = tl.load(Enabled + row)
    # Compare against float32(0.95), matching the sampler parameter tensor.
    normal = (
        enabled
        & (k == 20)
        & (p32 == 0.949999988079071)
        & (bad == 0)
        & (mlo > 0.0)
        & (mhi <= 2.0)
    )
    guarded = normal & (t19 > t20) & (t19 >= 2.0 ** (-100)) & (index < 20)
    first = tl.sum(tl.where(i == 0, t, 0.0), 0)
    second = tl.sum(tl.where(i == 1, t, 0.0), 0)
    single = normal & (first >= 2.0 ** (-100)) & (second == 0.0)
    chosen = tl.where(single, 0, tl.minimum(index, 0 + 31))
    threshold = tl.sum(tl.where(i == chosen, t, 0.0), 0)
    tl.store(Guard + row, guarded | single)
    tl.store(Onehot + row, single)
    tl.store(Cut + row, threshold)
    tl.store(Rank + row, tl.where(single, 1, index + 1))
    tl.store(Error + row, error)


def fused_certificate(torch, s, top32, top_ks, top_ps, enabled, buffers=None):
    assert s.ndim == 2 and s.is_contiguous() and (str(s.dtype) == "torch.float32")
    (rows, v) = s.shape
    assert top32.shape == (rows, 32) and top32.is_contiguous()
    chunks = triton.cdiv(v, 4096)
    if buffers is None:
        buffers = (
            torch.empty((rows, chunks), device=s.device, dtype=torch.float64),
            torch.empty((rows, chunks), device=s.device, dtype=torch.int32),
            torch.empty(rows, device=s.device, dtype=torch.bool),
            torch.empty(rows, device=s.device, dtype=torch.float32),
            torch.empty(rows, device=s.device, dtype=torch.float64),
            torch.empty(rows, device=s.device, dtype=torch.int32),
            torch.empty(rows, device=s.device, dtype=torch.bool),
        )
    (mass, bad, guard, cut, error, rank, onehot) = buffers
    _sum_parts[rows, chunks](
        s,
        mass,
        bad,
        V=v,
        CHUNKS=chunks,
        BLOCK=4096,
        num_warps=4,
        enable_fp_fusion=False,
        enable_reflect_ftz=False,
    )
    _certificate[rows,](
        top32,
        mass,
        bad,
        top_ks,
        top_ps,
        enabled,
        guard,
        cut,
        error,
        rank,
        onehot,
        V=v,
        CHUNKS=chunks,
        MASS_BLOCK=triton.next_power_of_2(chunks),
        **constants(v),
        num_warps=4,
        enable_fp_fusion=False,
        enable_reflect_ftz=False,
    )
    return (
        guard,
        cut,
        {"error_upper": error, "cut_rank": rank, "onehot": onehot},
        buffers,
    )
