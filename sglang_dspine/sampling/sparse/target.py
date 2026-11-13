"""Small-support joint P, retaining native softmax and top-p cutoff semantics.

The fast path needs a finite, host-known top-k bound and no min-p/greedy rows.
Unsupported batches return None; callers retain their ordinary dense builder.
Probabilities in the small prefix are *absolute full-vocabulary probabilities*.
They must not be softmaxed or renormalized before the top-p support test.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass
class SparseTarget:
    packed: tuple
    dense_p: torch.Tensor
    denominator: torch.Tensor | None = None


@triton.jit
def _pack_joint(TopIds, TopP, Nucleus, TopKs, Ids, Values, Lengths, Overflow,
                Threshold, ROW_WIDTH: tl.constexpr, CAP: tl.constexpr,
                BLOCK: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    k = tl.load(TopKs + row)
    threshold = tl.load(TopP + row * ROW_WIDTH + k - 1)
    p = tl.load(TopP + row * ROW_WIDTH + lane, lane < ROW_WIDTH, 0.)
    nucleus = tl.load(Nucleus + row * ROW_WIDTH + lane,
                      lane < ROW_WIDTH, 0.)
    ids = tl.load(TopIds + row * ROW_WIDTH + lane,
                  lane < ROW_WIDTH, 2147483647).to(tl.int32)
    keep = (lane < ROW_WIDTH) & (p > 0.) & (p >= threshold) & (nucleus > 0.)
    count = tl.sum(keep.to(tl.int32), 0)
    total = tl.sum(tl.where(keep, p, 0.), 0)
    # Sort by vocabulary ID to retain the existing CDF / uniform convention.
    # The deployed Triton exposes scalar sort rather than sort_key_value.
    # Pack the positive fp32 bits under the vocabulary-ID sort key.
    key_ids = tl.where(keep, ids, 2147483647).to(tl.int64)
    key_p = tl.where(keep, p, 0.).to(tl.int32, bitcast=True).to(tl.int64)
    ordered = tl.sort((key_ids << 32) | key_p, descending=False)
    sort_ids = (ordered >> 32).to(tl.int32)
    sort_p = (ordered & 0xFFFFFFFF).to(tl.int32).to(tl.float32, bitcast=True)
    tl.store(Ids + row * CAP + lane,
             tl.where(lane < count, sort_ids, -1), lane < CAP)
    tl.store(Values + row * CAP + lane,
             tl.where(lane < count, sort_p / total, 0.), lane < CAP)
    tl.store(Lengths + row, tl.minimum(count, CAP))
    tl.store(Overflow + row, count > CAP)
    tl.store(Threshold + row, threshold)


@triton.jit
def _overflow_dense(P, Overflow, Threshold, V: tl.constexpr,
                    BLOCK: tl.constexpr):
    row = tl.program_id(0)
    if tl.load(Overflow + row) != 0:
        # With k <= CAP, a retained (CAP+1)th item must tie the kth cutoff.
        # All p >= that cutoff therefore also satisfy the top-p support test.
        # Only overflow rows read/write V here; packed rows do no dense work.
        threshold = tl.load(Threshold + row)
        lane = tl.arange(0, BLOCK)
        total_lanes = tl.full((BLOCK,), 0., tl.float32)
        for start in range(0, V, BLOCK):
            i = start + lane
            value = tl.load(P + row * V + i, i < V, 0.)
            total_lanes += tl.where(value >= threshold, value, 0.)
        total = tl.sum(total_lanes, 0)
        for start in range(0, V, BLOCK):
            i = start + lane
            value = tl.load(P + row * V + i, i < V, 0.)
            value = tl.where(value >= threshold, value / total, 0.)
            tl.store(P + row * V + i, value, i < V)


def _row_indices(logits, info, row_to_request, row_offsets, block_size):
    rows = logits.shape[0]
    if row_to_request is not None:
        if row_to_request.numel() != rows:
            raise ValueError("row_to_request must cover every target row")
        return row_to_request.reshape(-1).to(device=logits.device, dtype=torch.long)
    index = torch.arange(rows, device=logits.device, dtype=torch.long)
    if row_offsets is not None:
        # No variable-sized repeat_interleave or device-to-host size query.
        return torch.bucketize(index, row_offsets[1:].contiguous(), right=True)
    bs = info.temperatures.numel()
    if rows != bs * block_size:
        raise ValueError("Compact target logits require row mapping")
    return index // block_size


def build_sparse_target(logits, info, *, row_to_request=None, row_offsets=None,
                        verify_lens=None, block_size=16, capacity=32,
                        sampling_backend="flashinfer"):
    """Return SparseTarget or None when the ordinary dense path is required.

    packed=(ids, values, lengths, overflow) has compact target-row indexing.
    Only overflow rows of dense_p contain normalized filtered P; other rows
    are intentionally unused by accept_sparse. No host read of CUDA tensors.
    verify_lens is accepted for the shared caller contract; row mapping alone
    determines this producer's rows.
    """
    max_k = getattr(info, "max_top_k", None)
    if (sampling_backend != "flashinfer"
            or getattr(info, "need_min_p_sampling", False)
            or getattr(info, "is_any_greedy", False)
            or not isinstance(max_k, int) or not 1 <= max_k <= capacity):
        return None
    if logits.ndim != 2 or capacity < 1:
        raise ValueError("Expected [rows,vocab] logits and positive capacity")
    rows, vocab = logits.shape
    if vocab < 1 or max_k > vocab:
        return None
    request = _row_indices(logits, info, row_to_request, row_offsets, block_size)
    if rows == 0:
        return SparseTarget((torch.empty((0, capacity), device=logits.device, dtype=torch.int32),
                             torch.empty((0, capacity), device=logits.device, dtype=torch.float32),
                             torch.empty(0, device=logits.device, dtype=torch.int32),
                             torch.empty(0, device=logits.device, dtype=torch.int32)),
                            torch.empty((0, vocab), device=logits.device, dtype=torch.float32))
    temperatures = info.temperatures.reshape(-1)[request]
    top_ks = info.top_ks.reshape(-1)[request].to(torch.int32).contiguous()
    top_ps = info.top_ps.reshape(-1)[request].to(torch.float32).contiguous()
    if logits.is_cuda:
        from sglang.kernels.ops.speculative.dspark.dspark_accept import SoftmaxTemp
        from flashinfer.sampling import top_p_renorm_probs
        p = SoftmaxTemp.execute(logits=logits, temperatures=temperatures,
                                rows_per_request=1).to(torch.float32)
    else:
        p = torch.softmax(logits.float() / temperatures.float()[:, None], dim=-1)
    width = min(capacity + 1, vocab)
    top_p, top_ids = torch.topk(p, width, dim=-1, sorted=True)
    if logits.is_cuda:
        # Native top-p sees absolute p. If the prefix mass is below top_p,
        # its cutoff is outside this prefix and all positive prefix items stay.
        # Padding avoids vector-tail assumptions in the native CUDA primitive.
        padded_width = max(triton.next_power_of_2(width), triton.next_power_of_2(capacity))
        small = torch.nn.functional.pad(top_p, (0, padded_width - width))
        nucleus = top_p_renorm_probs(small, top_ps)
        if padded_width != width:
            top_ids = torch.nn.functional.pad(top_ids, (0, padded_width - width), value=2147483647)
            top_p = small
        ids = torch.empty((rows, capacity), device=logits.device, dtype=torch.int32)
        values = torch.empty((rows, capacity), device=logits.device, dtype=torch.float32)
        lengths = torch.empty(rows, device=logits.device, dtype=torch.int32)
        overflow = torch.empty_like(lengths)
        threshold = torch.empty(rows, device=logits.device, dtype=torch.float32)
        _pack_joint[(rows,)](top_ids, top_p, nucleus, top_ks, ids, values,
                             lengths, overflow, threshold, ROW_WIDTH=padded_width,
                             CAP=capacity, BLOCK=padded_width)
        _overflow_dense[(rows,)](p, overflow, threshold, V=vocab, BLOCK=4096)
        return SparseTarget((ids, values, lengths, overflow), p)
    # CPU mathematical implementation for reference tests; no GPU behavior is
    # selected by these host-side branches.
    threshold = top_p.gather(1, (top_ks.long() - 1)[:, None])
    higher_mass = (top_p[:, None, :] *
                   (top_p[:, None, :] > top_p[:, :, None])).sum(-1)
    keep = ((top_p > 0) & (top_p >= threshold)
            & ((higher_mass < top_ps[:, None]) | (top_ps[:, None] >= 1)))
    count = keep.sum(-1)
    kept = top_p.masked_fill(~keep, 0)
    kept = kept / kept.sum(-1, keepdim=True)
    ordered, order = top_ids.masked_fill(~keep, 2147483647).sort(-1)
    kept = kept.gather(1, order)
    ids = torch.full((rows, capacity), -1, dtype=torch.int32, device=logits.device)
    values = torch.zeros((rows, capacity), dtype=torch.float32, device=logits.device)
    n = min(width, capacity)
    ids[:, :n] = ordered[:, :n].masked_fill(ordered[:, :n] == 2147483647, -1).int()
    values[:, :n] = kept[:, :n]
    overflow = (count > capacity).int()
    for row in range(rows):
        if overflow[row]:
            p[row] = p[row].masked_fill(p[row] < threshold[row], 0)
            p[row] /= p[row].sum()
    return SparseTarget((ids, values, count.clamp_max(capacity).int(), overflow), p)
