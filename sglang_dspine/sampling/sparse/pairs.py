"""DSpine inference kernels and graph-owned buffers."""

import triton
import triton.language as tl


@triton.jit
def _pairs(
    TopIds,
    TopVals,
    Modes,
    KCut,
    PCut,
    NativeK,
    NativeP,
    Ids,
    Weights,
    Counts,
    Sparse,
    Overflow,
    V: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, 32)
    ids = tl.load(TopIds + row * 32 + lane).to(tl.int32)
    val = tl.load(TopVals + row * 32 + lane)
    mode = tl.load(Modes + row)
    if mode == 0:
        kk = tl.load(NativeK + row * V + ids) > 0.0
        pp = tl.load(NativeP + row * V + ids)
    else:
        pp = val >= tl.load(PCut + row)
        kk = tl.full((32,), True, tl.int1)
        if mode == 1:
            kk = val >= tl.load(KCut + row)
    keep = kk & pp & (val > 0.0)
    count = tl.sum(keep.to(tl.int32), 0)
    rank = tl.sum(((ids[:, None] > ids[None, :]) & keep[None, :]).to(tl.int32), 1)
    tl.store(Ids + row * 32 + lane, -1)
    tl.store(Weights + row * 32 + lane, 0.0)
    tl.debug_barrier()
    tl.store(Ids + row * 32 + rank, ids, keep)
    tl.store(Weights + row * 32 + rank, val, keep)
    use = (mode != 0) & (count > 0) & (count < 32)
    tl.store(Counts + row, count)
    tl.store(Sparse + row, use)
    tl.store(Overflow + row, ~use)


@triton.jit
def _dense_fallback(
    S,
    Modes,
    KCut,
    PCut,
    NativeK,
    NativeP,
    Sparse,
    Out,
    V: tl.constexpr,
    BLOCK: tl.constexpr,
):
    (row, tile) = (tl.program_id(0), tl.program_id(1))
    if not tl.load(Sparse + row):
        i = tile * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(S + row * V + i, i < V, 0.0)
        mode = tl.load(Modes + row)
        if mode == 0:
            keep = (tl.load(NativeK + row * V + i, i < V, 0.0) > 0.0) & tl.load(
                NativeP + row * V + i, i < V, False
            )
        else:
            keep = x >= tl.load(PCut + row)
            if mode == 1:
                keep &= x >= tl.load(KCut + row)
        tl.store(Out + row * V + i, tl.where(keep, x, 0.0), i < V)


@triton.jit
def _divide(Weights, Denom, Probs):
    row = tl.program_id(0)
    i = tl.arange(0, 32)
    w = tl.load(Weights + row * 32 + i)
    d = tl.load(Denom + row)
    value = tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[w, d],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    tl.store(Probs + row * 32 + i, value)


def pairs(torch, values, ids, modes, kcut, pcut, kp, psupport):
    rows = values.shape[0]
    out_ids = torch.empty((rows, 32), device=values.device, dtype=torch.int32)
    weights = torch.empty_like(values)
    counts = torch.empty(rows, device=values.device, dtype=torch.int32)
    sparse = torch.empty(rows, device=values.device, dtype=torch.bool)
    overflow = torch.empty(rows, device=values.device, dtype=torch.int32)
    _pairs[rows,](
        ids,
        values,
        modes,
        kcut,
        pcut,
        kp,
        psupport,
        out_ids,
        weights,
        counts,
        sparse,
        overflow,
        V=kp.shape[-1],
        num_warps=1,
        enable_fp_fusion=False,
        enable_reflect_ftz=False,
    )
    return (out_ids, weights, counts, sparse, overflow)


def fallback(torch, s, modes, kcut, pcut, kp, psupport, sparse):
    out = torch.empty_like(s)
    _dense_fallback[s.shape[0], triton.cdiv(s.shape[1], 1024)](
        s,
        modes,
        kcut,
        pcut,
        kp,
        psupport,
        sparse,
        out,
        V=s.shape[1],
        BLOCK=1024,
        num_warps=4,
        enable_fp_fusion=False,
        enable_reflect_ftz=False,
    )
    return out


def divide(torch, weights, denom):
    out = torch.empty_like(weights)
    _divide[weights.shape[0],](
        weights,
        denom,
        out,
        num_warps=1,
        enable_fp_fusion=False,
        enable_reflect_ftz=False,
    )
    return out
