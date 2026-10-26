"""MSL ports of the four Triton code-bridge kernels.

The split mirrors ``sglang_dspine/kernels.py`` exactly: four kernels with the
three ``F.linear`` GEMMs left to MLX. Each kernel preserves the rounding points
of ``dspine_mlx.bridge.DSpineCodeBridge.__call__``:

* ``prepare_read``      → residuals / RMS / normalized
* ``normalize_codes``   → sqrt(C) scaling + anchor codebook substitution
* ``feature_message``   → sigmoid(Q[r] + S[r-1] + b) * V[r-1]
* ``feature_finish``    → bf16 write + fused Qwen3 RMSNorm (cast before affine)
"""

from __future__ import annotations

import math

import mlx.core as mx

from .common import as_f32, make_kernel

_F32_TINY = 1.1754943508222875e-38

_HEADER = r"""
#include <metal_stdlib>
using namespace metal;

static inline float tg_reduce_sum(float v, threadgroup float* scratch,
                                  uint simd_lane, uint simd_group) {
    float s = simd_sum(v);
    if (simd_lane == 0) scratch[simd_group] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = (simd_group == 0 && simd_lane < 8) ? scratch[simd_lane] : 0.0f;
    t = simd_sum(t);
    if (simd_group == 0 && simd_lane == 0) scratch[0] = t;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[0];
}
"""

_PREPARE = r"""
uint row = threadgroup_position_in_grid.x;
uint lane = thread_position_in_threadgroup.x;
uint sg = simdgroup_index_in_threadgroup;
uint sl = thread_index_in_simdgroup;

float acc = 0.0f;
float vals[PT];
for (int j = 0; j < PT; ++j) {
    int h = j * NT + (int)lane;
    if (h >= H) { vals[j] = 0.0f; continue; }
    float v = float(X[(size_t)row * H + h]);
    if (ADD) {
        v = float(bfloat16_t(v + float(Residual[(size_t)row * H + h])));
    }
    vals[j] = v;
    acc += v * v;
}
threadgroup float scratch[32];
float total = tg_reduce_sum(acc, scratch, sl, sg);
float rms = sqrt(fmax(total / (float)H, 1.1754943508222875e-38f));
for (int j = 0; j < PT; ++j) {
    int h = j * NT + (int)lane;
    if (h >= H) continue;
    Raw[(size_t)row * H + h] = bfloat16_t(vals[j]);
    Normalized[(size_t)row * H + h] = bfloat16_t(vals[j] / (rms + eps[0]));
}
if (lane == 0) RMS[row] = rms;
"""

_NORMALIZE = r"""
uint row = threadgroup_position_in_grid.x;
uint lane = thread_position_in_threadgroup.x;
uint sg = simdgroup_index_in_threadgroup;
uint sl = thread_index_in_simdgroup;
const float sqrt_c = sqrt((float)C);

if (ANCHOR_ROW && ((int)row % BT == 0)) {
    int token = anchors[row / BT];
    for (int j = 0; j < PT; ++j) {
        int c = j * NT + (int)lane;
        if (c >= C) continue;
        Codes[(size_t)row * C + c] =
            bfloat16_t(sqrt_c * float(Codebook[(size_t)token * C + c]));
    }
    return;
}

float acc = 0.0f;
for (int j = 0; j < PT; ++j) {
    int c = j * NT + (int)lane;
    if (c >= C) continue;
    float p = float(Projected[(size_t)row * C + c]);
    acc += p * p;
}
threadgroup float scratch[32];
float total = tg_reduce_sum(acc, scratch, sl, sg);
float norm = sqrt(total);
for (int j = 0; j < PT; ++j) {
    int c = j * NT + (int)lane;
    if (c >= C) continue;
    float p = float(Projected[(size_t)row * C + c]);
    Codes[(size_t)row * C + c] = bfloat16_t((sqrt_c * p) / (norm + eps[0]));
}
"""

_MESSAGE = r"""
uint row = threadgroup_position_in_grid.x;
uint lane = thread_position_in_threadgroup.x;
if ((int)lane >= R) return;
int prev = row > 0 ? (int)row - 1 : 0;
float q = float(Packed[(size_t)row * (3 * R) + lane]);
float s = float(Packed[(size_t)prev * (3 * R) + R + lane]);
float v = float(Packed[(size_t)prev * (3 * R) + 2 * R + lane]);
float gate = 1.0f / (1.0f + exp(-(q + s + Bias[lane])));
Message[(size_t)row * R + lane] =
    ((int)row % BT == 0) ? bfloat16_t(0.0f) : bfloat16_t(gate * v);
"""

_FINISH = r"""
uint row = threadgroup_position_in_grid.x;
uint lane = thread_position_in_threadgroup.x;
uint sg = simdgroup_index_in_threadgroup;
uint sl = thread_index_in_simdgroup;

float rms_v = RMS[row];
float acc = 0.0f;
bfloat16_t outs[PT];
for (int j = 0; j < PT; ++j) {
    int h = j * NT + (int)lane;
    if (h >= H) { outs[j] = bfloat16_t(0.0f); continue; }
    float r = float(Raw[(size_t)row * H + h]);
    float x = r;
    if (WRITE && ((int)row % BT != 0)) {
        x = r + rms_v * Delta[(size_t)row * H + h];
    }
    bfloat16_t ov = bfloat16_t(x);
    outs[j] = ov;
    Out[(size_t)row * H + h] = ov;
    float xr = float(ov);
    acc += xr * xr;
}
threadgroup float scratch[32];
float total = tg_reduce_sum(acc, scratch, sl, sg);
if (!NORM) return;
float inv = 1.0f / sqrt(total / (float)H + norm_eps[0]);
for (int j = 0; j < PT; ++j) {
    int h = j * NT + (int)lane;
    if (h >= H) continue;
    bfloat16_t y = bfloat16_t(float(outs[j]) * inv);
    Normed[(size_t)row * H + h] = bfloat16_t(float(y) * float(Weight[h]));
}
"""


_KERNELS: dict[str, object] = {}


def _get(name: str):
    if name in _KERNELS:
        return _KERNELS[name]
    if name == "prepare":
        k = mx.fast.metal_kernel(
            name="dspine_prepare_read",
            input_names=["X", "Residual", "eps"],
            output_names=["Raw", "Normalized", "RMS"],
            source=_PREPARE,
            header=_HEADER,
            compile_options={"math_mode": "safe"},
        )
    elif name == "normalize":
        k = mx.fast.metal_kernel(
            name="dspine_normalize_codes",
            input_names=["Projected", "Codebook", "anchors", "eps"],
            output_names=["Codes"],
            source=_NORMALIZE,
            header=_HEADER,
            compile_options={"math_mode": "safe"},
        )
    elif name == "message":
        k = mx.fast.metal_kernel(
            name="dspine_feature_message",
            input_names=["Packed", "Bias"],
            output_names=["Message"],
            source=_MESSAGE,
            header=_HEADER,
            compile_options={"math_mode": "safe"},
        )
    elif name == "finish":
        k = mx.fast.metal_kernel(
            name="dspine_feature_finish",
            input_names=["Raw", "Delta", "RMS", "Weight", "norm_eps"],
            output_names=["Out", "Normed"],
            source=_FINISH,
            header=_HEADER,
            compile_options={"math_mode": "safe"},
        )
    else:
        raise KeyError(name)
    _KERNELS[name] = k
    return k


def _per_thread(size: int, nt: int) -> int:
    return (size + nt - 1) // nt


def prepare_read(
    x: mx.array, residual: mx.array | None, eps: float, nt: int = 256
) -> tuple[mx.array, mx.array, mx.array]:
    m, h = x.shape
    add = residual is not None
    res = x if residual is None else residual
    raw, normalized, rms = _get("prepare")(
        inputs=[x, res, as_f32(eps)],
        template=[("H", int(h)), ("NT", int(nt)), ("PT", _per_thread(h, nt)),
                  ("ADD", bool(add))],
        grid=(m * nt, 1, 1),
        threadgroup=(nt, 1, 1),
        output_shapes=[(m, h), (m, h), (m,)],
        output_dtypes=[x.dtype, x.dtype, mx.float32],
    )
    return raw, normalized, rms


def normalize_codes(
    projected: mx.array,
    codebook: mx.array,
    anchors: mx.array,
    eps: float,
    *,
    block_size: int,
    nt: int = 256,
) -> mx.array:
    m, c = projected.shape
    (codes,) = _get("normalize")(
        inputs=[projected, codebook, anchors.astype(mx.int32), as_f32(eps)],
        template=[("C", int(c)), ("BT", int(block_size)), ("NT", int(nt)),
                  ("PT", _per_thread(c, nt)), ("ANCHOR_ROW", True)],
        grid=(m * nt, 1, 1),
        threadgroup=(nt, 1, 1),
        output_shapes=[(m, c)],
        output_dtypes=[projected.dtype],
    )
    return codes


def feature_message(
    packed: mx.array, bias: mx.array, *, block_size: int, nt: int = 512
) -> mx.array:
    m, r3 = packed.shape
    r = r3 // 3
    (message,) = _get("message")(
        inputs=[packed, bias.astype(mx.float32)],
        template=[("R", int(r)), ("BT", int(block_size))],
        grid=(m * nt, 1, 1),
        threadgroup=(nt, 1, 1),
        output_shapes=[(m, r)],
        output_dtypes=[packed.dtype],
    )
    return message


def feature_finish(
    raw: mx.array,
    delta: mx.array,
    rms: mx.array,
    weight: mx.array,
    norm_eps: float,
    *,
    block_size: int,
    write: bool = True,
    norm: bool = True,
    nt: int = 256,
) -> tuple[mx.array, mx.array]:
    m, h = raw.shape
    out, normed = _get("finish")(
        inputs=[raw, delta, rms, weight, as_f32(norm_eps)],
        template=[("H", int(h)), ("BT", int(block_size)), ("NT", int(nt)),
                  ("PT", _per_thread(h, nt)), ("WRITE", bool(write)),
                  ("NORM", bool(norm))],
        grid=(m * nt, 1, 1),
        threadgroup=(nt, 1, 1),
        output_shapes=[(m, h), (m, h)],
        output_dtypes=[raw.dtype, raw.dtype],
    )
    return out, normed


def bridge_call(
    bridge,
    hidden: mx.array,
    residual: mx.array | None,
    input_ids: mx.array,
    layer: int,
    next_norm,
    stash: dict | None = None,
) -> tuple[mx.array, mx.array]:
    """Drop-in replacement for ``DSpineCodeBridge.__call__`` (four MSL kernels)."""
    if hidden.dtype != mx.bfloat16:
        raise ValueError("the MSE bridge requires BF16 activations")
    b, t, h = hidden.shape
    m = b * t
    sqrt_c = math.sqrt(bridge.code_size)

    raw, normalized, rms = prepare_read(
        hidden.reshape(m, h),
        None if residual is None else residual.reshape(m, h),
        bridge.eps,
    )
    projected = mx.matmul(normalized, bridge.layer_read_weight(layer).T)
    codes = normalize_codes(
        projected,
        bridge.codebook,
        input_ids[:, 0],
        bridge.eps,
        block_size=t,
    )
    packed = mx.matmul(codes, bridge.packed_weight.T)
    if stash is not None:
        stash.update(
            raw=raw,
            rms=rms,
            packed=packed,
        )
    message = feature_message(packed, bridge.gate.bias, block_size=t)
    delta = mx.matmul(message, bridge.feature_write_weights[layer].T).astype(mx.float32)
    out, normed = feature_finish(
        raw,
        delta,
        rms,
        next_norm.weight,
        next_norm.eps,
        block_size=t,
    )
    return normed.reshape(b, t, h), out.reshape(b, t, h)
