"""DSpine inference kernels and graph-owned buffers."""

import triton
import triton.language as tl


@triton.jit
def _mask(
    Rows, ActualBS, Effective, M: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    real = offsets // H < tl.load(ActualBS) * 15
    value = tl.load(Rows + offsets, (offsets < M * H) & real, 0.0)
    tl.store(Effective + offsets, value, offsets < M * H)


def mask(rows, actual_bs, out):
    (m, h) = rows.shape
    assert rows.is_contiguous() and out.shape == rows.shape and out.is_contiguous()
    _mask[triton.cdiv(m * h, 512),](
        rows, actual_bs, out, M=m, H=h, BLOCK=512, num_warps=4
    )
    return out
