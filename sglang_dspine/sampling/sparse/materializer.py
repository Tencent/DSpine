"""DSpine inference kernels and graph-owned buffers."""

import triton
import triton.language as tl


@triton.jit
def _fill_dense(Raw, Denom, Sparse, Out, V: tl.constexpr, BLOCK: tl.constexpr):
    (row, tile) = (tl.program_id(0), tl.program_id(1))
    i = tile * BLOCK + tl.arange(0, BLOCK)
    valid = i < V
    if tl.load(Sparse + row):
        tl.store(Out + row * V + i, 0.0, valid)
    else:
        raw = tl.load(Raw + row * V + i, valid, 0.0)
        denominator = tl.load(Denom + row)
        p = tl.inline_asm_elementwise(
            "div.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[raw, denominator],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        tl.store(Out + row * V + i, p, valid)


@triton.jit
def _scatter_sparse(Ids, Probs, Lengths, Sparse, Out, V: tl.constexpr):
    row = tl.program_id(0)
    if tl.load(Sparse + row):
        lane = tl.arange(0, 32)
        count = tl.load(Lengths + row)
        ids = tl.load(Ids + row * 32 + lane)
        valid = (lane < count) & (ids >= 0) & (ids < V)
        p = tl.load(Probs + row * 32 + lane, valid, 0.0)
        tl.store(Out + row * V + ids, p, valid)


def materialize(torch, bundle):
    """Return a new contiguous FP32 [R,V] using P4's original p bits.

    No sync, host data transfer, nonzero, index_select, RNG or global stream
    selection occurs here. The fill and scatter launches follow the caller's
    current CUDA stream, including nondefault streams and graph capture.
    """
    (ids, probs) = (bundle["ids"], bundle["probs"])
    (lengths, sparse) = (bundle["lengths"], bundle["use_sparse"])
    (raw, denominator) = (bundle["raw_masked"], bundle["denominator"])
    (rows, vocab) = raw.shape
    assert 16 <= rows <= 128 and rows % 16 == 0 and (vocab == 248320)
    assert ids.shape == probs.shape == (rows, 32)
    assert lengths.shape == sparse.shape == denominator.shape == (rows,)
    assert ids.dtype == lengths.dtype == torch.int32 and sparse.dtype == torch.bool
    assert probs.dtype == raw.dtype == denominator.dtype == torch.float32
    for tensor in (ids, probs, lengths, sparse, raw, denominator):
        assert (
            tensor.is_cuda and tensor.is_contiguous() and (tensor.device == raw.device)
        )
    out = torch.empty((rows, vocab), device=raw.device, dtype=torch.float32)
    _fill_dense[rows, triton.cdiv(vocab, 1024)](
        raw,
        denominator,
        sparse,
        out,
        V=vocab,
        BLOCK=1024,
        num_warps=4,
        enable_fp_fusion=False,
        enable_reflect_ftz=False,
    )
    _scatter_sparse[rows,](
        ids,
        probs,
        lengths,
        sparse,
        out,
        V=vocab,
        num_warps=1,
        enable_fp_fusion=False,
        enable_reflect_ftz=False,
    )
    return out
