"""DSpine inference kernels and graph-owned buffers."""

import triton
import triton.language as tl


@triton.jit
def gather_dot(
    W,
    IDs,
    Hidden,
    Out,
    H: tl.constexpr,
    C: tl.constexpr,
    BC: tl.constexpr,
    BH: tl.constexpr,
):
    (row, group) = (tl.program_id(0), tl.program_id(1))
    cs = group * BC + tl.arange(0, BC)
    tokens = tl.load(IDs + row * C + cs, cs < C, 0).to(tl.int64)
    hs = tl.arange(0, BH)
    acc = tl.full((BC, BH), 0.0, tl.float32)
    for block in tl.static_range((H + BH - 1) // BH):
        k = block * BH + hs
        hidden = tl.load(Hidden + row * H + k, k < H, 0).to(tl.float32)
        weights = tl.load(
            W + tokens[:, None] * H + k[None, :],
            (cs[:, None] < C) & (k[None, :] < H),
            0,
        ).to(tl.float32)
        acc = acc + weights * hidden[None, :]
    score = tl.sum(acc, 1)
    tl.store(Out + row * C + cs, score, cs < C)


@triton.jit
def global_top16(Scores, IDs, Out, C: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, C)
    values = tl.load(Scores + row * C + i).to(tl.float32)
    tokens = tl.load(IDs + row * C + i).to(tl.int32)
    lane = tl.arange(0, 16)
    chosen = tl.zeros((16,), tl.int32)
    for j in tl.static_range(16):
        largest = tl.max(values, 0)
        token = tl.min(tl.where(values == largest, tokens, 2147483647), 0)
        chosen = tl.where(lane == j, token, chosen)
        values = tl.where(tokens == token, float("-inf"), values)
    tl.store(Out + row * 16 + lane, tl.sort(chosen).to(tl.int64))


def rescore(torch, weight, ids, hidden, *, fp32_output=False):
    (m, c) = ids.shape
    h = hidden.shape[1]
    out = torch.empty(
        (m, c),
        device=hidden.device,
        dtype=torch.float32 if fp32_output else hidden.dtype,
    )
    gather_dot[m, triton.cdiv(c, 4)](
        weight,
        ids,
        hidden,
        out,
        H=h,
        C=c,
        BC=4,
        BH=1024,
        num_warps=4,
        enable_fp_fusion=False,
        enable_reflect_ftz=False,
    )
    return out


def select16(torch, scores, ids):
    out = torch.empty((ids.shape[0], 16), device=ids.device, dtype=torch.int64)
    global_top16[ids.shape[0],](scores, ids, out, C=ids.shape[1], num_warps=4)
    return out
