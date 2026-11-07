"""DSpine inference kernels and graph-owned buffers."""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from sglang_dspine.sampling.sparse.support import constants


@triton.jit
def _ksafe(
    Top,
    Mass,
    Bad,
    Ks,
    Enabled,
    PGuard,
    Mode,
    Cut,
    V: tl.constexpr,
    C: tl.constexpr,
    B: tl.constexpr,
    GM: tl.constexpr,
):
    row = tl.program_id(0)
    j = tl.arange(0, B)
    mh = tl.sum(tl.load(Mass + row * C + j, j < C, 0.0), 0)
    bad = tl.sum(tl.load(Bad + row * C + j, j < C, 0), 0)
    inf = tl.full((), float("inf"), tl.float64)
    ninf = -inf
    one = tl.full((), 1.0, tl.float64)
    gm = tl.full((), GM, tl.float64)
    lo = libdevice.nextafter(mh / libdevice.nextafter(one + gm, inf), ninf)
    hi = libdevice.nextafter(mh / libdevice.nextafter(one - gm, ninf), inf)
    pivot = tl.load(Top + row * 32 + 19)
    normal = (
        tl.load(Enabled + row)
        & (tl.load(Ks + row) == 20)
        & (bad == 0)
        & (lo > 0.0)
        & (hi <= 2.0)
    )
    safe_pivot = normal & (pivot >= 2.0 ** (-100))
    p_cert = tl.load(PGuard + row)
    mode = tl.where(safe_pivot, 1, tl.where(p_cert, 2, 0))
    tl.store(Mode + row, mode)
    tl.store(Cut + row, pivot)


def k_modes(torch, s, top32, ks, enabled, p_guard, certificate_buffers):
    (rows, v) = s.shape
    (mass, bad) = certificate_buffers[:2]
    chunks = mass.shape[1]
    mode = torch.empty(rows, device=s.device, dtype=torch.int32)
    cut = torch.empty(rows, device=s.device, dtype=torch.float32)
    _ksafe[rows,](
        top32,
        mass,
        bad,
        ks,
        enabled,
        p_guard,
        mode,
        cut,
        V=v,
        C=chunks,
        B=triton.next_power_of_2(chunks),
        GM=constants(v)["GM"],
        num_warps=4,
        enable_fp_fusion=False,
        enable_reflect_ftz=False,
    )
    return (mode, cut)
