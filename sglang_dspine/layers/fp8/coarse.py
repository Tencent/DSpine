"""Row-scaled FP8 retrieval; final scores use the original BF16 head."""


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
    def __init__(self, torch, original_weight):
        self.torch = torch
        self.original_weight = original_weight
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
        scale = _scale(torch, hidden)
        quantized = _quantize(torch, hidden, scale)
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
        ids = fi_top_k(coarse, 128, sorted=False, dsa_graph_safe=True)[1]
        scores = gather_ops.rescore(torch, self.original_weight, ids, hidden)
        return gather_ops.select16(torch, scores, ids), ids, scores
