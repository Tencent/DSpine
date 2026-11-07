"""Exact joint top-k/top-p support with the original FP32 reduction."""

import threading

_air = _k = _sum = None
_buffers = {}


def initialize():
    global _air, _k, _sum
    if _air is None:
        from sglang_dspine.kernels.extensions import sampling_kernels

        (_air, _k, _sum) = sampling_kernels()


def from_probs(torch, s, ks, ps, *, enabled=None):
    from flashinfer.topk import top_k

    import sglang_dspine.sampling.sparse.pairs as pair_kernels
    from sglang_dspine.sampling.sparse.support import fused_certificate
    from sglang_dspine.sampling.sparse.topk_guard import k_modes

    assert _air is not None
    (rows, vocab) = s.shape
    assert vocab == 248320 and rows % 16 == 0 and (16 <= rows <= 128)
    (values, ids) = top_k(s, 32, sorted=False, dsa_graph_safe=True)
    order = values.argsort(-1, descending=True)
    values = values.gather(-1, order).contiguous()
    ids = ids.gather(-1, order).contiguous()
    if enabled is None:
        enabled = torch.ones(rows, device=s.device, dtype=torch.bool)
    (certified, cut, detail, massbuf) = fused_certificate(
        torch, s, values, ks, ps, enabled
    )
    (mode, kcut) = k_modes(torch, s, values, ks, enabled, certified, massbuf)
    dispatch_cert = certified & (mode != 0)
    support = torch.empty_like(s, dtype=torch.bool)
    pcut = torch.empty(rows, device=s.device, dtype=torch.float32)
    align = lambda x: (x + 255) // 256 * 256
    workspace = torch.empty(
        align(384 * rows) + 2 * align(4 * 2048 * rows) + 2 * align(4 * vocab * rows),
        device=s.device,
        dtype=torch.uint8,
    )
    visits = torch.empty((rows, 3), device=s.device, dtype=torch.int32)
    _air.sparse_air_cut_out(
        s, ps, dispatch_cert, cut, mode, pcut, support, workspace, visits, False
    )
    key = (
        rows,
        str(s.device),
        threading.get_ident(),
        int(torch.cuda.current_stream(s.device).cuda_stream),
    )
    if key not in _buffers:
        _buffers[key] = (
            torch.zeros(rows * 4096, device=s.device, dtype=torch.uint8),
            torch.empty_like(s),
            torch.empty(rows, device=s.device, dtype=torch.int32),
        )
    (states, kp, kvisits) = _buffers[key]
    _k.conditional_topk_out(s, ks, mode != 0, kp, states, kvisits, False)
    (packed_ids, weights, counts, use_sparse, overflow) = pair_kernels.pairs(
        torch, values, ids, mode, kcut, pcut, kp, support
    )
    raw = pair_kernels.fallback(torch, s, mode, kcut, pcut, kp, support, use_sparse)
    partials = torch.empty((rows, 31), device=s.device, dtype=torch.float32)
    denom = torch.empty(rows, device=s.device, dtype=torch.float32)
    _sum.mixed_sum_out(packed_ids, weights, counts, use_sparse, raw, partials, denom)
    probs = pair_kernels.divide(torch, weights, denom)
    return dict(
        ids=packed_ids,
        weights=weights,
        probs=probs,
        lengths=counts,
        use_sparse=use_sparse,
        overflow=overflow,
        raw_masked=raw,
        denominator=denom,
        certified=certified,
        dispatch_certified=dispatch_cert,
        k_modes=mode,
        visits=visits,
        k_visits=kvisits,
        p_cut=pcut,
        k_cut=kcut,
    )


def build_sparse_joint(*, logits, info, bs, width, native=None, **kw):
    import torch

    if native is None:
        from sglang_dspine.runtime.worker import build_dspine_target_probs as native
    assert width == 16 and 1 <= bs <= 8
    assert (
        info.need_top_k_sampling
        and info.need_top_p_sampling
        and (not info.need_min_p_sampling)
    )
    s = (
        native.__globals__["SoftmaxTemp"]
        .execute(logits=logits, temperatures=info.temperatures, rows_per_request=width)
        .to(torch.float32)
    )
    ks = torch.repeat_interleave(info.top_ks.view(-1), width)
    ps = torch.repeat_interleave(info.top_ps.view(-1), width)
    return from_probs(torch, s, ks, ps, **kw)
