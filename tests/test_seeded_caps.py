"""Dynamic sampled allocation requires independent explicit request streams."""

from collections import Counter
from types import SimpleNamespace

import pytest

from sglang_dspine.verification.caps import VerifyCaps
from sglang_dspine.verification.policy import SpsPolicy
from sglang_dspine.verification.ragged import RaggedVerifyCaps


def _controller(ragged=False):
    cls = RaggedVerifyCaps if ragged else VerifyCaps
    controller = object.__new__(cls)
    controller.full = 16
    controller.sampled_dynamic = True
    controller.counts = Counter()
    controller.policy = (
        SpsPolicy(token_budgets={3: 24})
        if ragged else SpsPolicy(verify_widths={3: 12})
    )
    if ragged:
        controller.ragged_backend = object()
        controller.ragged_graph = None
    return controller


def _batch(seeds):
    return SimpleNamespace(
        batch_size=lambda: len(seeds),
        sampling_info=SimpleNamespace(is_all_greedy=False, is_any_greedy=False),
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        has_grammar=False,
        return_logprob=False,
        reqs=[SimpleNamespace(sampling_params=SimpleNamespace(sampling_seed=seed))
              for seed in seeds],
    )


@pytest.mark.parametrize("ragged", [False, True])
@pytest.mark.parametrize("seeds", [[7, 7, 19], [-3, 5, -3], [0, 0, None]])
def test_repeated_explicit_seed_uses_full_block_without_mutating_requests(ragged, seeds):
    controller = _controller(ragged)
    batch = _batch(seeds)
    controller.begin(batch)
    assert controller.sampled_round
    assert not controller.in_round
    assert controller.active == 16
    assert not controller.counts
    assert [req.sampling_params.sampling_seed for req in batch.reqs] == seeds
    if ragged:
        assert not controller.ragged_round
        assert controller.batch_size == 0


@pytest.mark.parametrize("ragged", [False, True])
@pytest.mark.parametrize(
    "seeds", [[7, 8, 19], [7, 7 + 2**32, -(2**63)], [None, None, None], [7, None, None]]
)
def test_distinct_64bit_seeds_and_unseeded_requests_remain_dynamic(ragged, seeds):
    controller = _controller(ragged)
    batch = _batch(seeds)
    controller.begin(batch)
    assert controller.in_round
    assert controller.active < 16
    assert [req.sampling_params.sampling_seed for req in batch.reqs] == seeds
    if ragged:
        assert controller.ragged_round


def test_effective_seed_matches_the_full_64bit_hash_representation():
    controller = _controller()
    batch = _batch([-1, 2**64 - 1, 19])
    controller.begin(batch)
    assert not controller.in_round
    assert controller.active == 16


def test_mixed_greedy_sampling_keeps_the_existing_full_block_rule():
    controller = _controller(True)
    batch = _batch([7, 8, 9])
    batch.sampling_info.is_any_greedy = True
    controller.begin(batch)
    assert not controller.in_round
    assert controller.active == 16
    assert not controller.ragged_round


def test_missing_request_list_preserves_cpu_fixture_behavior():
    controller = _controller()
    batch = _batch([7, 7, 7])
    del batch.reqs
    controller.begin(batch)
    assert controller.in_round
    assert controller.active == 12
