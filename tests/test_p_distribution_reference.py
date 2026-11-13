"""Independent native-AR references for compact target probabilities."""

from types import SimpleNamespace

import pytest
import torch


def _row_reference(logits, temperature, top_k, top_p, min_p=0.0,
                   *, minp_branch=False, greedy=False):
    """Explicit float64 reference on unique scores; native GPU handles cutoff ties."""
    if greedy:
        return torch.nn.functional.one_hot(logits.argmax(), logits.numel()).double()
    p = torch.softmax(logits.double() / float(temperature), -1)
    order = torch.argsort(-p, stable=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(p.numel(), device=p.device)
    keep_k = rank < int(top_k)
    base = torch.where(keep_k, p, 0.0) if minp_branch else p
    if minp_branch:
        base = base / base.sum()
    sorted_p = base[order]
    before = sorted_p.cumsum(0) - sorted_p
    keep_p = torch.empty_like(keep_k)
    keep_p[order] = before <= float(top_p)
    keep = keep_k & keep_p
    if minp_branch or min_p:
        keep &= base >= float(min_p) * base.max()
    selected = torch.where(keep, p, 0.0)
    return selected / selected.sum()


def _info(bs, device="cpu", *, mixed=False, minp=False, top_k=20):
    return SimpleNamespace(
        temperatures=torch.linspace(.7, 1.4, bs, device=device).view(-1, 1),
        top_ks=torch.tensor(([1, 5, 20] * bs)[:bs] if mixed else [top_k] * bs,
                            dtype=torch.int32, device=device),
        top_ps=torch.linspace(.65, .95, bs, device=device),
        min_ps=torch.linspace(0., .15, bs, device=device) if minp else torch.zeros(bs, device=device),
        need_top_k_sampling=True,
        need_top_p_sampling=True,
        need_min_p_sampling=minp,
        is_any_greedy=mixed,
        is_all_greedy=False,
        max_top_k=max(20, top_k) if mixed else top_k,
    )


def _layout(bs, device="cpu"):
    lens = torch.tensor(([1, 2, 8, 16] * bs)[:bs], dtype=torch.int32, device=device)
    offsets = torch.cat((torch.zeros(1, device=device, dtype=torch.int32), lens.cumsum(0).int()))
    requests = torch.repeat_interleave(torch.arange(bs, device=device, dtype=torch.int32), lens)
    return lens, offsets[:-1].contiguous(), requests


def test_joint_nucleus_uses_absolute_full_vocabulary_mass():
    logits = torch.tensor([.4, .3, .2, .1]).log()
    joint = _row_reference(logits, 1., 2, .5)
    conditional = _row_reference(logits, 1., 2, .5, minp_branch=True)
    torch.testing.assert_close(joint, torch.tensor([4 / 7, 3 / 7, 0, 0]).double())
    assert conditional.tolist() == [1., 0., 0., 0.]


def test_zero_minp_row_still_obeys_batch_minp_branch():
    logits = torch.tensor([.4, .3, .2, .1]).log()
    p = _row_reference(logits, 1., 2, .5, 0., minp_branch=True)
    assert p.tolist() == [1., 0., 0., 0.]


@pytest.mark.parametrize("bs", [1, 3, 9, 17, 32, 33, 65])
def test_compact_reference_maps_each_row_to_its_request(bs):
    lens, offsets, requests = _layout(bs)
    info = _info(bs, mixed=True)
    logits = torch.linspace(-2., 2., 97)
    rows = [_row_reference(logits, info.temperatures[b], info.top_ks[b], info.top_ps[b],
                           greedy=bool(info.top_ks[b] <= 1)) for b in requests]
    assert len(rows) == int(lens.sum())
    for b, offset in enumerate(offsets.tolist()):
        assert int(requests[offset]) == b
        assert int(requests[offset + int(lens[b]) - 1]) == b
        torch.testing.assert_close(rows[offset].sum(), torch.tensor(1.).double())
    # Shape-only repeat_interleave(block16) would select the wrong request here.
    if bs > 1:
        assert int(requests[1]) == 1
        assert not torch.equal(rows[0], rows[1])


def test_mixed_greedy_ties_use_first_argmax_id():
    p = _row_reference(torch.zeros(97), .7, 1, .9, greedy=True)
    assert p[0] == 1 and p[1:].count_nonzero() == 0


@pytest.mark.parametrize("bs", [1, 9, 33])
def test_cpu_sparse_producer_matches_independent_joint_reference(bs):
    from sglang_dspine.sampling.sparse.target import build_sparse_target

    torch.manual_seed(1701 + bs)
    lens, offsets, requests = _layout(bs)
    info = _info(bs)
    logits = torch.randn((requests.numel(), 97)) * 2
    result = build_sparse_target(logits, info, row_to_request=requests,
                                 row_offsets=offsets, verify_lens=lens)
    assert result is not None
    expected = torch.stack([_row_reference(row, info.temperatures[b],
        info.top_ks[b], info.top_ps[b]) for row, b in zip(logits, requests)]).float()
    _reconstruct(result, expected)
    from_offsets = build_sparse_target(logits, info, row_offsets=offsets, verify_lens=lens)
    _reconstruct(from_offsets, expected)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _native_reference(logits, info, request_ids, backend="flashinfer"):
    """Native primitives only; never calls the candidate sparse producer."""
    from sglang.kernels.ops.speculative.dspark.dspark_accept import SoftmaxTemp
    from sglang.srt.speculative.dflash_utils import (
        _dflash_top_k_renorm_prob, _dflash_top_p_renorm_prob,
    )

    req = request_ids.long()
    temperature = info.temperatures.index_select(0, req)
    top_k = info.top_ks.index_select(0, req)
    top_p = info.top_ps.index_select(0, req)
    min_p = info.min_ps.index_select(0, req)
    p = SoftmaxTemp.execute(logits=logits, temperatures=temperature, rows_per_request=1).float()
    if backend == "flashinfer":
        if info.need_min_p_sampling:
            p = _dflash_top_k_renorm_prob(p, top_k)
            p = _dflash_top_p_renorm_prob(p, top_p)
            p = p.masked_fill(p < p.amax(-1, keepdim=True) * min_p[:, None], 0.)
        else:
            keep = torch.ones_like(p, dtype=torch.bool)
            if info.need_top_k_sampling:
                keep &= _dflash_top_k_renorm_prob(p, top_k) > 0
            if info.need_top_p_sampling:
                keep &= _dflash_top_p_renorm_prob(p, top_p) > 0
            p = p.masked_fill(~keep, 0.)
    else:
        # Match the native PyTorch backend's sorted weights explicitly.
        values, ids = p.sort(-1, descending=True)
        cumulative = values.cumsum(-1)
        values = values.masked_fill(torch.arange(p.shape[-1], device=p.device)[None] >= top_k[:, None], 0.)
        values = values.masked_fill(cumulative - values > top_p[:, None], 0.)
        if info.need_min_p_sampling:
            values = values.masked_fill(values < values[:, :1] * min_p[:, None], 0.)
        p = torch.zeros_like(p).scatter_(-1, ids, values)
    p = p / p.sum(-1, keepdim=True)
    if info.is_any_greedy:
        greedy = top_k <= 1
        onehot = torch.zeros_like(p).scatter_(-1, logits.argmax(-1, keepdim=True), 1.)
        p = torch.where(greedy[:, None], onehot, p)
    return p


def _reconstruct(result, expected):
    ids, values, lengths, overflow = result.packed
    rows, capacity = ids.shape
    assert rows == expected.shape[0]
    assert values.shape == ids.shape and lengths.shape == overflow.shape == (rows,)
    valid = torch.arange(capacity, device=ids.device)[None] < lengths[:, None]
    assert bool(((lengths >= 0) & (lengths <= capacity)).all())
    assert bool(((ids >= 0) & (ids < expected.shape[-1]) | ~valid).all())
    assert bool(torch.isfinite(values).all()) and bool((values >= 0).all())
    if capacity > 1:
        assert bool(((ids[:, 1:] > ids[:, :-1]) | ~valid[:, 1:]).all())
    p = torch.zeros_like(expected).scatter_add_(-1, ids.long().clamp_min(0), values * valid)
    if bool(overflow.any()):
        assert result.dense_p is not None, "Overflow cannot truncate target mass without a dense fallback"
        assert result.dense_p.shape == expected.shape
        p = torch.where(overflow[:, None].bool(), result.dense_p, p)
    assert torch.equal(p > 0, expected > 0)
    torch.testing.assert_close(p, expected, rtol=2e-5, atol=2e-7)
    torch.testing.assert_close(p.sum(-1), torch.ones(rows, device=p.device), rtol=2e-6, atol=2e-6)
    return p


@cuda
@pytest.mark.parametrize("bs", [1, 3, 9, 17, 32, 33, 65])
@pytest.mark.parametrize("mixed", [False, True])
def test_sparse_target_matches_native_joint_for_compact_mixed_rows(bs, mixed):
    from sglang_dspine.sampling.sparse.target import build_sparse_target

    torch.manual_seed(1801 + bs)
    lens, offsets, requests = _layout(bs, "cuda")
    info = _info(bs, "cuda", mixed=mixed)
    logits = (torch.randn((requests.numel(), 257), device="cuda") * 2).bfloat16()
    p = _native_reference(logits, info, requests)
    result = build_sparse_target(logits, info, row_to_request=requests,
                                 row_offsets=offsets, verify_lens=lens, block_size=16)
    if mixed:
        assert result is None, "Mixed greedy uses the exact native fallback"
    else:
        assert result is not None, "The supported finite-K joint path must be exercised"
        _reconstruct(result, p)


@cuda
@pytest.mark.parametrize("kind", ["cutoff_ties", "nucleus_ties", "overflow", "unlimited", "minp", "pytorch"])
def test_sparse_target_uses_exact_native_fallback_on_boundary_cases(kind):
    from sglang_dspine.sampling.sparse.target import build_sparse_target

    bs = 9
    lens, offsets, requests = _layout(bs, "cuda")
    info = _info(bs, "cuda", mixed=True, minp=kind == "minp")
    logits = torch.randn((requests.numel(), 97), device="cuda")
    if kind == "cutoff_ties":
        logits.fill_(-3.)
        logits[:, 0], logits[:, 1:7] = 3., 2.
        info.top_ks.fill_(5)
        info.max_top_k = 5
        info.top_ps.fill_(.95)
        info.is_any_greedy = False
    if kind == "overflow":
        logits.zero_()
        info.top_ks.fill_(20)
        info.top_ps.fill_(1.)
        info.is_any_greedy = False
    if kind == "nucleus_ties":
        logits.fill_(-100.)
        logits[:, :4] = torch.tensor([.5, .25, .125, .125], device="cuda").log()
        info.top_ks.fill_(20)
        info.top_ps.fill_(.75)
        info.is_any_greedy = False
    if kind == "unlimited":
        info.top_ks.fill_(1 << 30)
        info.max_top_k = 1 << 30
        info.top_ps.fill_(1.)
        info.need_top_k_sampling = info.need_top_p_sampling = False
    backend = "pytorch" if kind == "pytorch" else "flashinfer"
    p = _native_reference(logits, info, requests, backend)
    result = build_sparse_target(logits, info, row_to_request=requests, row_offsets=offsets,
                                 verify_lens=lens, sampling_backend=backend, capacity=32)
    if result is not None:
        _reconstruct(result, p)
    # None is the documented exact-native fallback; integration is checked below.


@cuda
@pytest.mark.parametrize("vocab", [65537, 248320])
def test_sparse_target_covers_large_vocabulary_and_bf16_logits(vocab):
    from sglang_dspine.sampling.sparse.target import build_sparse_target

    info = _info(3, "cuda")
    lens = torch.tensor([1, 2, 3], device="cuda", dtype=torch.int32)
    offsets = torch.tensor([0, 1, 3], device="cuda", dtype=torch.int32)
    requests = torch.tensor([0, 1, 1, 2, 2, 2], device="cuda", dtype=torch.int32)
    logits = (torch.randn((6, vocab), device="cuda") * 3).bfloat16()
    p = _native_reference(logits, info, requests)
    result = build_sparse_target(logits, info, row_to_request=requests,
                                 row_offsets=offsets, verify_lens=lens)
    assert result is not None
    _reconstruct(result, p)


@cuda
@pytest.mark.parametrize("bs", [3, 9, 32])
@pytest.mark.parametrize("reverse_packed", [False, True])
def test_compact_packed_rejection_matches_dense_reference_with_identical_coins(bs, reverse_packed):
    from sglang_dspine.sampling.rejection import sample_capped_chain_from_probs
    from sglang_dspine.sampling.sparse.accept import accept_sparse
    from sglang_dspine.sampling.sparse.target import build_sparse_target

    torch.manual_seed(1907 + bs)
    width, vocab, k = 16, 257, 16
    lens, offsets, requests = _layout(bs, "cuda")
    info = _info(bs, "cuda")
    logits = torch.randn((requests.numel(), vocab), device="cuda") * 2
    p = _native_reference(logits, info, requests)
    result = build_sparse_target(logits, info, row_to_request=requests,
                                 row_offsets=offsets, verify_lens=lens)
    assert result is not None
    _reconstruct(result, p)
    packed = result.packed
    if reverse_packed:
        ids, values, lengths, overflow = packed
        order = torch.where(torch.arange(ids.shape[1], device="cuda")[None] < lengths[:, None],
                            lengths[:, None] - 1 - torch.arange(ids.shape[1], device="cuda")[None],
                            torch.arange(ids.shape[1], device="cuda")[None]).long()
        packed = ids.gather(1, order), values.gather(1, order), lengths, overflow
    dense_p = torch.zeros((bs, width, vocab), device="cuda")
    dense_p[..., 0] = 1.
    q_ids = torch.empty((bs, width - 1, k), device="cuda", dtype=torch.int64)
    for b in range(bs):
        start, count = int(offsets[b]), int(lens[b])
        dense_p[b, :count] = p[start:start + count]
        for depth in range(width - 1):
            q_ids[b, depth] = p[start + min(depth, count - 1)].topk(k).indices.sort().values
    q_values = torch.softmax(torch.randn((bs, width - 1, k), device="cuda"), -1)
    chosen = torch.multinomial(q_values.reshape(-1, k), 1).view(bs, width - 1, 1)
    draft = q_ids.gather(-1, chosen).squeeze(-1)
    candidates = torch.cat((torch.zeros((bs, 1), device="cuda", dtype=torch.int64), draft), 1)
    dense_q = torch.zeros((bs, width - 1, vocab), device="cuda").scatter_(-1, q_ids, q_values)
    coins, bonus_coins = torch.rand((bs, width - 1), device="cuda"), torch.rand(bs, device="cuda")
    for b in range(0, bs, 2):
        for depth in range(width - 1):
            candidates[b, depth + 1] = p[int(offsets[b]) + min(depth, int(lens[b]) - 1)].argmax()
        coins[b].zero_()  # Exercise every cap and its compact terminal bonus row.
    prefix = torch.arange(bs, device="cuda", dtype=torch.int32) + 8190
    def buffers():
        return (torch.empty(bs, device="cuda", dtype=torch.int32),
                torch.empty(bs, device="cuda", dtype=torch.int32),
                torch.empty(bs, device="cuda", dtype=torch.int64),
                torch.empty((bs, width), device="cuda", dtype=torch.int64),
                torch.empty(bs, device="cuda", dtype=torch.int32))
    actual, expected = buffers(), buffers()
    saved = q_values.clone(), coins.clone(), bonus_coins.clone(), torch.cuda.get_rng_state().clone()
    accept_sparse(torch, candidates=candidates, dense_p=result.dense_p, packed=packed,
                  q_ids=q_ids, q_values=q_values, verify_lens=lens,
                  uniforms=coins, uniform_final=bonus_coins, prefix_lens=prefix,
                  outputs=actual, row_offsets=offsets, vocab_size=vocab)
    sample_capped_chain_from_probs(candidates=candidates, target_probs=dense_p,
        draft_probs=dense_q, verify_lens=lens, uniforms=coins, uniform_final=bonus_coins,
        prefix_lens=prefix, outputs=expected)
    assert all(torch.equal(a, e) for a, e in zip(actual, expected))
    assert torch.equal(actual[0][::2], lens[::2] - 1)
    assert torch.equal(saved[0], q_values) and torch.equal(saved[1], coins)
    assert torch.equal(saved[2], bonus_coins) and torch.equal(saved[3], torch.cuda.get_rng_state())


@pytest.mark.parametrize("custom,mask,eligible", [(False, None, True), (True, None, False),
                                                (False, [], False), (True, [], False)])
def test_compact_eligibility_retains_dense_envelope_for_custom_or_mask_capture(custom, mask, eligible):
    from sglang_dspine.sampling.compact import can_keep_compact

    info = SimpleNamespace(has_custom_logit_processor=custom, sampling_mask_batch_indices=mask)
    assert can_keep_compact(info) is eligible
    assert can_keep_compact(None) is False


@cuda
@pytest.mark.parametrize("kind", ["mixed", "minp", "unlimited", "pytorch"])
def test_target_fallback_selects_sampling_params_by_compact_request_index(kind, monkeypatch):
    from sglang_dspine.sampling.target import TargetDistribution
    from sglang_dspine.sampling.sparse import accept
    from sglang.srt import runtime_context
    from sglang_dspine.sampling import random

    bs = 9
    lens, offsets, requests = _layout(bs, "cuda")
    info = _info(bs, "cuda", mixed=kind == "mixed", minp=kind == "minp")
    if kind == "unlimited":
        info.max_top_k = 1 << 30
        info.top_ks.fill_(1 << 30)
        info.top_ps.fill_(1.)
        info.need_top_k_sampling = info.need_top_p_sampling = False
    backend = "pytorch" if kind == "pytorch" else "flashinfer"
    logits = torch.randn((requests.numel(), 97), device="cuda")
    expected = _native_reference(logits, info, requests, backend)
    outputs, seen = object(), {}
    sampler = SimpleNamespace(round_sampled=True, round_bs=bs, k=16, random=None,
        candidate_out=torch.arange(16, device="cuda").expand(bs, 15, 16).contiguous(),
        q_out=torch.full((bs, 15, 16), 1 / 16, device="cuda"))
    worker = SimpleNamespace(block_size=16, _dspine_sampler=sampler,
                             _next_accept_bonus_buffers=lambda size: outputs)
    monkeypatch.setattr(runtime_context, "get_exec", lambda: SimpleNamespace(
        kernel=SimpleNamespace(sampling_backend=backend)))
    monkeypatch.setattr(random, "verification_uniforms", lambda *args: (
        torch.full((bs, 15), .5, device="cuda"), torch.full((bs,), .5, device="cuda")))
    monkeypatch.setattr(accept, "accept_sparse", lambda *args, **kwargs: seen.update(kwargs))
    original = tuple(getattr(info, name).clone() for name in ("temperatures", "top_ks", "top_ps", "min_ps"))
    result = TargetDistribution(worker).sample_prefix(
        candidates=torch.zeros((bs, 16), device="cuda", dtype=torch.int64),
        next_token_logits=logits, sampling_info=info,
        prefix_lens=torch.zeros(bs, device="cuda", dtype=torch.int32), bs=bs,
        verify_lens=lens, row_offsets=offsets, row_to_request=requests.long())
    assert result is outputs and seen["row_offsets"] is offsets
    assert torch.equal(seen["dense_p"] > 0, expected > 0)
    torch.testing.assert_close(seen["dense_p"], expected, rtol=2e-5, atol=2e-7)
    assert all(torch.equal(saved, getattr(info, name)) for saved, name in zip(
        original, ("temperatures", "top_ks", "top_ps", "min_ps")))


@cuda
@pytest.mark.parametrize("kind", ["linear_bias", "live_penalizer", "grammar"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compact_logit_adjustments_match_dense_native_hook(kind, dtype, monkeypatch):
    from sglang.srt.constrained.base_grammar_backend import GrammarMask
    from sglang.srt.sampling.penaltylib.orchestrator import BatchedPenalizerOrchestrator
    from sglang.srt.sampling.penaltylib.repetition_penalty import BatchedRepetitionPenalizer
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
    from sglang_dspine.sampling.compact import around_verify_logits_adjustments
    from sglang_dspine.verification import caps

    bs, width, vocab = 3, 16, 97
    lens, offsets, requests = _layout(bs, "cuda")
    depth = torch.cat([torch.arange(int(n), device="cuda") for n in lens])
    flat_ids = requests.long() * width + depth
    info = object.__new__(SamplingBatchInfo)
    info.temperatures = torch.ones(bs, device="cuda")
    info.acc_additive_penalties = torch.randn((bs, vocab), device="cuda")
    info.acc_scaling_penalties = .5 + torch.rand((bs, vocab), device="cuda") * 2
    info.logit_bias = torch.randn((bs, vocab), device="cuda")
    if kind == "live_penalizer":
        orch = object.__new__(BatchedPenalizerOrchestrator)
        orch.is_required = True
        repetition = BatchedRepetitionPenalizer(orch)
        repetition._is_prepared = True
        repetition.cumulated_repetition_penalties = info.acc_scaling_penalties
        orch.penalizers = {type(repetition): repetition}
        info.penalizer_orchestrator = orch
        info.acc_scaling_penalties = None
    elif kind == "grammar":
        class BooleanGrammar:
            def apply_vocab_mask(self, *, logits, vocab_mask):
                logits.masked_fill_(~vocab_mask, -torch.inf)
        info.grammar_mask = GrammarMask(BooleanGrammar(),
            torch.rand(bs, vocab, device="cuda") > .3)
    logits = torch.randn((bs * width, vocab), device="cuda").to(dtype)
    expected, actual = logits.clone(), logits.index_select(0, flat_ids).contiguous()
    for position in range(width):
        info.apply_logits_bias(expected.view(bs, width, vocab)[:, position])
    controller = SimpleNamespace(compact_round=True, compact_logits=actual,
                                 row_to_request=requests.long())
    monkeypatch.setattr(caps, "active_controller", lambda: controller)
    def unexpected_dense(**kwargs):
        raise AssertionError("Compact rows must not enter the native B*16 shape assertion")
    around_verify_logits_adjustments(unexpected_dense, next_token_logits=actual,
        sampling_info=info, draft_token_num=width)
    assert torch.equal(actual, expected.index_select(0, flat_ids))
    controller.compact_round = False
    fallback = logits.clone()
    around_verify_logits_adjustments(unexpected_dense,
        next_token_logits=fallback, sampling_info=info, draft_token_num=width)
    assert torch.equal(fallback, expected)
