"""Row-scaled FP8 retrieval; final scores use the original BF16 head."""

import torch


def stable_local_topk(scores, k, fi_top_k):
    """Use FlashInfer's deterministic lower-index boundary tie selection."""
    if scores.device.type != "cuda":
        ids = torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :k]
        return scores.gather(-1, ids), ids
    values, ids = fi_top_k(
        scores,
        k,
        sorted=False,
        deterministic=True,
        tie_break=1,
        dsa_graph_safe=True,
    )
    return values, ids.long()


def _scale(torch, values):
    maximum = values.float().abs().amax(dim=1, keepdim=True)
    return (
        torch.where(maximum > 0, maximum / 448.0, torch.ones_like(maximum))
        .float()
        .contiguous()
    )


def _quantize(torch, values, scale):
    return (
        (values.float() / scale)
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )


class FullVFP8Coarse:
    def __init__(self, torch, original_weight, tp_head=None):
        self.torch = torch
        self.original_weight = original_weight
        self.tp_head = tp_head
        vocab, hidden = original_weight.shape
        self.weight8 = torch.empty(
            (vocab, hidden), device=original_weight.device, dtype=torch.float8_e4m3fn
        )
        scales = torch.empty(vocab, device=original_weight.device, dtype=torch.float32)
        for start in range(0, vocab, 4096):
            end = min(vocab, start + 4096)
            block = original_weight[start:end]
            scale = _scale(torch, block)
            self.weight8[start:end].copy_(_quantize(torch, block, scale))
            scales[start:end].copy_(scale[:, 0])
        self.weight_scale = scales.reshape(1, vocab).contiguous()
        self.weight8_t = self.weight8.T

    def retrieve_and_rescore(self, hidden, *, fi_top_k, gather_ops):
        torch = self.torch
        scale = _scale(self.torch, hidden)
        quantized = _quantize(self.torch, hidden, scale)
        coarse = torch.ops.aten._scaled_mm.default(
            quantized,
            self.weight8_t,
            scale,
            self.weight_scale,
            bias=None,
            scale_result=None,
            out_dtype=torch.bfloat16,
            use_fast_accum=False,
        )
        if self.tp_head is None:
            ids = fi_top_k(coarse, 128, sorted=False, dsa_graph_safe=True)[1]
            scores = gather_ops.rescore(torch, self.original_weight, ids, hidden)
        else:
            head = self.tp_head
            coarse[:, head.num_local :].fill_(float("-inf"))
            width = min(128, coarse.shape[1])
            values, ids = stable_local_topk(coarse, width, fi_top_k)
            ids = ids + head.start
            values = values.masked_fill(ids >= head.end, float("-inf"))
            ids = ids.masked_fill(ids >= head.end, head.global_vocab_size)
            # All ranks exchange 128 slots, including shards smaller than 128.
            if width < 128:
                values = torch.nn.functional.pad(
                    values, (0, 128 - width), value=float("-inf")
                )
                ids = torch.nn.functional.pad(
                    ids, (0, 128 - width), value=head.global_vocab_size
                )
            _, ids = head.global_topk(values, ids, 128)
            scores = head.rescore(ids, hidden)
        return gather_ops.select16(torch, scores, ids), ids, scores
