"""Shared helpers for the hand-written MSL experiment kernels."""

from __future__ import annotations

import mlx.core as mx

# MLX concatenates ``header`` directly in front of the generated
# ``template <...>`` declaration, so it must end with a newline.
HEADER = r"""
#include <metal_stdlib>
using namespace metal;
"""


def make_kernel(name: str, input_names, output_names, source: str):
    return mx.fast.metal_kernel(
        name=name,
        input_names=list(input_names),
        output_names=list(output_names),
        source=source,
        header=HEADER,
        ensure_row_contiguous=True,
        compile_options={"math_mode": "safe"},
    )


def as_f32(x) -> mx.array:
    return mx.array([float(x)], dtype=mx.float32)


def as_i32(x) -> mx.array:
    return mx.array([int(x)], dtype=mx.int32)
