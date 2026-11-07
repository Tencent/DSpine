"""Cached factorized proposal probabilities and exact cancellation fallback."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from . import cancellation as P

build_cache = P.build_cache


@triton.jit
def _prepare_x(
    Raw, RMS, X, Scale, First, H: tl.constexpr, T: tl.constexpr, BH: tl.constexpr
):
    n = tl.program_id(0)
    source = n // (T - 1) * T + n % (T - 1) + 1
    h = tl.arange(0, BH)
    x = tl.load(Raw + source * H + h, h < H, 0).to(tl.float64)
    a = tl.load(RMS + source).to(tl.float64)
    first = tl.sum(x * x, axis=0)
    tl.store(X + n * H + h, x, h < H)
    tl.store(Scale + n, a)
    tl.store(First + n, first)


@triton.jit
def _prepare_message(
    Query,
    Anchors,
    Cand,
    G,
    V,
    TG,
    Bias,
    Message,
    TGc,
    R: tl.constexpr,
    K: tl.constexpr,
    T: tl.constexpr,
    BR: tl.constexpr,
):
    nj = tl.program_id(0)
    n = nj // K
    j = nj % K
    request = n // (T - 1)
    position = n % (T - 1)
    if position == 0:
        sid = tl.load(Anchors + request)
    else:
        sid = tl.load(Cand + (n - 1) * K + j)
    cid = tl.load(Cand + nj)
    source = request * T + position + 1
    r = tl.arange(0, BR)
    q = tl.load(Query + source * R + r, r < R, 0).to(tl.float64)
    g = tl.load(G + sid * R + r, r < R, 0).to(tl.float64)
    v = tl.load(V + sid * R + r, r < R, 0).to(tl.float64)
    bias = tl.load(Bias + r, r < R, 0).to(tl.float64)
    gate = q + g + bias
    sigmoid = 1.0 / (1.0 + libdevice.exp(-gate))
    message = sigmoid * v
    tg = tl.load(TG + cid * R + r, r < R, 0).to(tl.float64)
    tl.store(Message + nj * R + r, message, r < R)
    tl.store(TGc + nj * R + r, tg, r < R)


@triton.jit
def _finalize(
    Message,
    XW,
    MG,
    Scale,
    First,
    Base,
    Product,
    Safe,
    Force,
    Out,
    D,
    Mag,
    R: tl.constexpr,
    K: tl.constexpr,
    H: tl.constexpr,
    EPS: tl.constexpr,
    USE_FORCE: tl.constexpr,
    DIAGNOSTIC: tl.constexpr,
    BR: tl.constexpr,
    BK: tl.constexpr,
):
    nj = tl.program_id(0)
    n = nj // K
    r = tl.arange(0, BR)
    m = tl.load(Message + nj * R + r, r < R, 0).to(tl.float64)
    xw = tl.load(XW + n * R + r, r < R, 0).to(tl.float64)
    mg = tl.load(MG + nj * R + r, r < R, 0).to(tl.float64)
    cross = tl.sum(xw * m, axis=0)
    quadratic = tl.sum(mg * m, axis=0)
    a = tl.load(Scale + n).to(tl.float64)
    first = tl.load(First + n).to(tl.float64)
    second = 2.0 * a * cross
    third = a * a * quadratic
    d = first + second + third
    magnitude = tl.abs(first) + tl.abs(second) + tl.abs(third)
    safe = (
        libdevice.isfinited(d)
        & libdevice.isfinited(magnitude)
        & (d >= 0)
        & ((magnitude == 0) | (d > 2.0 ** (-24) * magnitude))
    )
    if USE_FORCE:
        safe = safe & ~tl.load(Force + nj)
    tl.store(Safe + nj, safe)
    denominator = libdevice.sqrt_rn(d / H + EPS)
    c = tl.arange(0, BK)
    base = tl.load(Base + n * K + c, c < K, 0).to(tl.float64)
    product = tl.load(Product + nj * K + c, c < K, 0).to(tl.float64)
    numerator = base + a * product
    output = (numerator / denominator).to(tl.float32)
    tl.store(Out + nj * K + c, output, c < K)
    if DIAGNOSTIC:
        tl.store(D + nj, d)
        tl.store(Mag + nj, magnitude)


@dataclass
class Workspace:
    x: torch.Tensor
    scale: torch.Tensor
    first: torch.Tensor
    message: torch.Tensor
    tg: torch.Tensor
    safe: torch.Tensor
    d: torch.Tensor
    magnitude: torch.Tensor
    fallback: object

    @classmethod
    def allocate(cls, n, k, h, r, device):
        f = lambda shape: torch.empty(shape, device=device, dtype=torch.float64)
        return cls(
            f((n, h)),
            f((n,)),
            f((n,)),
            f((n, k, r)),
            f((n, k, r)),
            torch.empty((n, k), device=device, dtype=torch.bool),
            f((n, k)),
            f((n, k)),
            P.Workspace.allocate(n, k, h, device),
        )

    def validate(self, n, k, h, r, device):
        expected = {
            "x": (n, h),
            "scale": (n,),
            "first": (n,),
            "message": (n, k, r),
            "tg": (n, k, r),
            "safe": (n, k),
            "d": (n, k),
            "magnitude": (n, k),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            assert (
                value.shape == shape
                and value.is_contiguous()
                and (value.device == torch.device(device))
            ), name
            assert value.dtype == (torch.bool if name == "safe" else torch.float64), (
                name
            )
        self.fallback.validate(n, k, h, device)


def factor_table(
    raw, rms, query, anchors, cand, cache, eps, t, out, base_out, workspace
):
    """Compute conditional proposal logits with FP64 factors and direct fallback."""
    (n, k) = cand.shape
    h = raw.shape[1]
    r = query.shape[1]
    bs = anchors.numel()
    assert raw.device.type == "cuda" and t >= 2 and (n == bs * (t - 1))
    assert (
        raw.shape == (bs * t, h)
        and rms.shape == (bs * t,)
        and (query.shape == (bs * t, r))
    )
    assert anchors.shape == (bs,) and anchors.dtype == cand.dtype == torch.int64
    assert all(
        (z.is_contiguous() for z in (raw, rms, query, anchors, cand, out, base_out))
    )
    assert out.shape == (n, k, k) and out.dtype == torch.float32
    assert base_out.shape == (n, k) and base_out.dtype == torch.float64
    assert cache["W"].shape == (h, r) and cache["Gram"].shape == (r, r)
    assert all(
        (
            cache[name].dtype == torch.float64
            for name in ("W", "Gram", "G", "V", "TG", "bias", "norm")
        )
    )
    workspace.validate(n, k, h, r, raw.device)
    _prepare_x[n,](
        raw,
        rms,
        workspace.x,
        workspace.scale,
        workspace.first,
        H=h,
        T=t,
        BH=triton.next_power_of_2(h),
        num_warps=8 if h >= 1024 else 4,
        enable_fp_fusion=False,
    )
    _prepare_message[n * k,](
        query,
        anchors,
        cand,
        cache["G"],
        cache["V"],
        cache["TG"],
        cache["bias"],
        workspace.message,
        workspace.tg,
        R=r,
        K=k,
        T=t,
        BR=triton.next_power_of_2(r),
        num_warps=4,
        enable_fp_fusion=False,
    )
    P.F._candidate_base[n * k,](
        workspace.x,
        cache["head"],
        cache["norm"],
        cand,
        base_out,
        K=k,
        H=h,
        BH=triton.next_power_of_2(h),
        num_warps=8,
    )
    product = torch.einsum("njr,ncr->njc", workspace.message, workspace.tg)
    xw = workspace.x @ cache["W"]
    mg = workspace.message @ cache["Gram"]
    _finalize[n * k,](
        workspace.message,
        xw,
        mg,
        workspace.scale,
        workspace.first,
        base_out,
        product,
        workspace.safe,
        workspace.safe,
        out,
        workspace.d,
        workspace.magnitude,
        R=r,
        K=k,
        H=h,
        EPS=float(eps),
        USE_FORCE=False,
        DIAGNOSTIC=False,
        BR=triton.next_power_of_2(r),
        BK=triton.next_power_of_2(k),
        num_warps=4,
        enable_fp_fusion=False,
    )
    P.apply_direct_fallback(
        workspace.x,
        workspace.scale,
        workspace.message,
        cand,
        cache,
        workspace.safe,
        out,
        workspace.fallback,
        eps,
    )
    torch._assert_async(
        torch.isfinite(out).all(),
        "Fused FP64 proposal nonfinite after direct64 fallback",
    )
    return out
