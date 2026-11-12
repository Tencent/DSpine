"""Large decode batches keep exact proposals and bypass batch-eight optimizations."""

from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang_dspine.sampling.factorized.graph import ProposalGraphs
from sglang_dspine.sampling.target import TargetDistribution
from sglang_dspine.verification.policy import SpsPolicy


@pytest.mark.parametrize("bs", [9, 16, 17, 32])
@pytest.mark.parametrize("greedy", [True, False])
def test_large_batches_verify_full_blocks(bs, greedy):
    from sglang_dspine.verification.ragged import RaggedVerifyCaps

    controller = object.__new__(RaggedVerifyCaps)
    controller.policy = SpsPolicy((12,) * 7, 64)
    controller.full = 16
    controller.tokens = 64
    controller.sampled_dynamic = True
    controller.counts = Counter()
    batch = SimpleNamespace(
        batch_size=lambda: bs,
        sampling_info=SimpleNamespace(is_all_greedy=greedy, is_any_greedy=greedy),
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        has_grammar=False,
        return_logprob=False,
    )
    controller.begin(batch)
    assert controller.in_round and controller.active == 16
    assert not controller.ragged_round


def test_large_target_sampling_bypasses_small_batch_sparse_path():
    distribution = TargetDistribution(None)
    distribution.certificate = object()
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        is_extend_in_batch=False,
        has_grammar=False,
        return_logprob=False,
        sampling_info=SimpleNamespace(),
        reqs=[None] * 9,
    )
    distribution.begin(batch)
    assert distribution.certificate is None
    assert (
        distribution.sample_prefix(
            candidates=None,
            next_token_logits=None,
            sampling_info=None,
            prefix_lens=None,
            bs=9,
            verify_lens=None,
        )
        is None
    )


def test_large_replayssm_forward_uses_native_implementation():
    from sglang_dspine.runtime.replayssm import hooks

    runner = object()
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_target_verify=lambda: True), batch_size=32
    )
    native = Mock(return_value=object())
    with (
        patch.object(hooks, "_pool", return_value=object()),
        patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
        patch.object(hooks.replay_guard, "launch") as guard,
    ):
        assert hooks.around_forward(native, runner, batch) is native.return_value
    native.assert_called_once_with(runner, batch)
    guard.assert_not_called()


def test_large_proposal_fallback_keeps_cache_integrity_checks():
    cache = object.__new__(ProposalGraphs)
    cache.sampler = SimpleNamespace(max_bs=32)
    source = torch.zeros(1)
    cache.source_tensors = (source,)
    from sglang_dspine.sampling.factorized.graph import _signature

    cache.source_signature = (_signature(source),)
    source.add_(1)
    with pytest.raises(RuntimeError, match="weights changed"):
        cache.lookup(9)
    with pytest.raises(ValueError, match="capacity"):
        cache.lookup(33)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("capacity", [4, 32])
def test_proposal_graphs_and_large_eager_batches_match_direct_refinement(capacity):
    torch.manual_seed(42)
    h, r, c, v, k = 64, 8, 16, 64, 16
    rand = lambda *shape: torch.randn(shape, device="cuda") * 0.1
    bridge = SimpleNamespace(codebook=rand(v, c).bfloat16())
    sampler = SimpleNamespace(
        max_bs=capacity,
        draft_model=SimpleNamespace(selected_prefix_bridge=bridge),
        pred_gate_f32=rand(r, c),
        prev_f32=rand(r, c),
        gate_b_f32=rand(r),
        write_f32=rand(h, r),
        norm_f32=torch.ones(h, device="cuda"),
        lm_head_weight=rand(v, h).bfloat16(),
        norm_eps=1e-6,
        raw=rand(capacity * 16, h).bfloat16(),
        rms=torch.ones(capacity * 16, device="cuda"),
        query=rand(capacity * 16, r).bfloat16(),
        anchors=torch.randint(v, (capacity,), device="cuda"),
        candidate_out=torch.randint(v, (capacity, 15, k), device="cuda"),
    )
    cache = ProposalGraphs(sampler)
    assert set(cache.entries) == set(range(1, min(8, capacity) + 1))
    # Exercise both transitions and different tail chunks; mutate live inputs
    # after capture so replay cannot pass by returning startup's cached values.
    for bs in (1, 4, 8, 9, 16, 17, 32, 9, 8, 4, 1):
        if bs > capacity:
            continue
        sampler.raw.uniform_(-0.1, 0.1)
        sampler.anchors.random_(v)
        actual = cache.lookup(bs)
        n = bs * 15
        candidates = sampler.candidate_out[:bs].reshape(n, k)
        source = torch.arange(n, device="cuda")
        source = source // 15 * 16 + source % 15 + 1
        predecessor = sampler.candidate_out[:bs].clone()
        predecessor[:, 1:] = sampler.candidate_out[:bs, :-1]
        predecessor[:, 0] = sampler.anchors[:bs, None]
        predecessor = predecessor.reshape(n, k)
        message = (
            torch.sigmoid(
                sampler.query[source, None].double()
                + cache.weights["G"][predecessor]
                + cache.weights["bias"]
            )
            * cache.weights["V"][predecessor]
        )
        state = sampler.raw[source, None].double() + message @ cache.weights["W"].T
        normalized = state / torch.sqrt(state.square().mean(-1, keepdim=True) + 1e-6)
        normalized *= cache.weights["norm"]
        expected = normalized @ sampler.lm_head_weight[candidates].double().transpose(
            1, 2
        )
        torch.testing.assert_close(actual, expected.float(), rtol=2e-5, atol=2e-6)
