"""DSpine serving on SGLang 0.5.22."""

import torch
import triton
import triton.language as tl


@triton.jit
def _packed_conv(
    X,
    Q,
    State,
    W,
    Bias,
    Slots,
    ScratchSlots,
    Windows,
    Y,
    D: tl.constexpr,
    T: tl.constexpr,
    XS0: tl.constexpr,
    XS1: tl.constexpr,
    SS0: tl.constexpr,
    SS1: tl.constexpr,
    SS2: tl.constexpr,
    WS0: tl.constexpr,
    WS1: tl.constexpr,
    IS0: tl.constexpr,
    IS1: tl.constexpr,
    IS2: tl.constexpr,
    IS3: tl.constexpr,
    BIAS: tl.constexpr,
    SILU: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    f = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(Slots + req).to(tl.int64)
    if slot < 0:
        return
    scratch = tl.load(ScratchSlots + req).to(tl.int64)
    start = tl.load(Q + req)
    length = tl.load(Q + req + 1) - start
    state_base = State + slot * SS0 + f * SS1
    c0 = tl.load(state_base, f < D, 0.0)
    c1 = tl.load(state_base + SS2, f < D, 0.0)
    c2 = tl.load(state_base + 2 * SS2, f < D, 0.0)
    w0 = tl.load(W + f * WS0, f < D, 0.0)
    w1 = tl.load(W + f * WS0 + WS1, f < D, 0.0)
    w2 = tl.load(W + f * WS0 + 2 * WS1, f < D, 0.0)
    w3 = tl.load(W + f * WS0 + 3 * WS1, f < D, 0.0)
    if BIAS:
        preload = tl.load(Bias + f, f < D, 0.0).to(tl.float32)
    else:
        preload = tl.zeros((BLOCK,), tl.float32)
    for t in tl.static_range(T):
        x = tl.load(X + (start + t) * XS0 + f * XS1, (t < length) & (f < D), 0.0)
        acc = preload
        acc += c0 * w0
        acc += c1 * w1
        acc += c2 * w2
        acc += x * w3
        c0 = c1
        c1 = c2
        c2 = x
        window = Windows + scratch * IS0 + t * IS1 + f * IS2
        tl.store(window, c0, (t < length) & (f < D))
        tl.store(window + IS3, c1, (t < length) & (f < D))
        tl.store(window + 2 * IS3, c2, (t < length) & (f < D))
        if SILU:
            acc = acc / (1 + tl.exp(-acc))
        tl.store(Y + (start + t) * D + f, acc, (t < length) & (f < D))
    tl.store(state_base, c0, f < D)
    tl.store(state_base + SS2, c1, f < D)
    tl.store(state_base + 2 * SS2, c2, f < D)


def packed_conv(
    x, indptr, state, weight, bias, activation, slots, scratch_slots, windows, steps=16
):
    if steps != 16 or weight.shape != (x.shape[-1], 4) or state.shape[-1] != 3:
        raise ValueError(
            "Packed convolution supports Qwen GDN width4, state3 and draft16"
        )
    if activation not in (None, False, True, "silu", "swish"):
        raise ValueError(f"Unsupported convolution activation {activation}")
    if x.dtype != torch.bfloat16 or state.dtype != x.dtype or weight.dtype != x.dtype:
        raise ValueError("Packed convolution requires native BF16 tensors")
    result = torch.empty_like(x, memory_format=torch.contiguous_format)
    dim = x.shape[-1]
    bs = indptr.numel() - 1
    _packed_conv[bs, triton.cdiv(dim, 256)](
        x,
        indptr,
        state,
        weight,
        bias,
        slots,
        scratch_slots,
        windows,
        result,
        dim,
        steps,
        *x.stride(),
        *state.stride(),
        *weight.stride(),
        *windows.stride(),
        bias is not None,
        activation in (True, "silu", "swish"),
        256,
        num_warps=4,
    )
    return result
