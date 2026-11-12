"""Request RNG reproducibility across batch layouts, positions and draw phases."""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from ref_utils import sampling_info

from sglang_dspine.sampling.random import (
    ACCEPT,
    BONUS,
    DRAFT,
    PREFILL,
    RequestRNG,
    _uniforms,
    sample_prefill,
    seeded_batch,
    seeded_uniforms,
    verification_uniforms,
)


@pytest.mark.parametrize("stream", [DRAFT, ACCEPT, BONUS, PREFILL])
def test_reordering_compaction_and_other_requests_do_not_change_draws(stream):
    seeds = torch.tensor([0, -1, -(2**63), 2**63 - 1, 2**40 + 7])
    positions = torch.tensor([0, 19, 214, 8192, 131071])
    state = torch.get_rng_state().clone()
    reference = seeded_uniforms(seeds, positions, 15, stream)
    torch.rand(97)
    assert torch.equal(reference, seeded_uniforms(seeds, positions, 15, stream))
    order = torch.tensor([4, 1, 3])
    assert torch.equal(
        reference[order], seeded_uniforms(seeds[order], positions[order], 15, stream)
    )
    assert torch.equal(
        reference[:1], seeded_uniforms(seeds[:1], positions[:1], 15, stream)
    )
    assert bool(((reference >= 0) & (reference < 1)).all())
    torch.set_rng_state(state)


def test_domains_positions_and_all_seed_bits_affect_the_draws():
    seeds, positions = torch.tensor([7]), torch.tensor([99])
    values = [seeded_uniforms(seeds, positions, 15, stream) for stream in range(4)]
    assert all(
        not torch.equal(values[i], values[j]) for i in range(4) for j in range(i)
    )
    assert not torch.equal(
        values[0], seeded_uniforms(seeds + 2**32, positions, 15, DRAFT)
    )
    assert not torch.equal(values[0], seeded_uniforms(seeds, positions + 1, 15, DRAFT))


def test_seeded_verification_is_independent_of_device_rng_and_previous_draws():
    random = RequestRNG()
    random.stage(torch.tensor([7, 123]), torch.tensor([29, 811]))
    before = torch.get_rng_state().clone()
    first = verification_uniforms(2, 16, "cpu", random)
    assert torch.equal(before, torch.get_rng_state())
    random.draw(15, DRAFT)
    second = verification_uniforms(2, 16, "cpu", random)
    assert all(torch.equal(x, y) for x, y in zip(first, second))
    assert not torch.equal(first[0][:, :1], first[1][:, None])


def test_unseeded_verification_retains_the_existing_rng_sequence():
    before = torch.get_rng_state().clone()
    expected = torch.rand(2, 15), torch.rand(2)
    torch.set_rng_state(before)
    actual = verification_uniforms(2, 16, "cpu", RequestRNG())
    assert all(torch.equal(x, y) for x, y in zip(actual, expected))


def _batch(seeds):
    return SimpleNamespace(
        reqs=[
            SimpleNamespace(sampling_params=SimpleNamespace(sampling_seed=s))
            for s in seeds
        ],
        sampling_info=SimpleNamespace(
            sampling_seed=None, temperatures=torch.ones(len(seeds), 1)
        ),
    )


def test_request_seed_scope_restores_scheduler_state_even_on_error():
    batch = _batch([0, None, -17])
    with pytest.raises(RuntimeError, match="test"):
        with seeded_batch(batch):
            first = batch.sampling_info.sampling_seed.clone()
            assert first[0] == 0 and first[2] == -17
            raise RuntimeError("test")
    assert batch.sampling_info.sampling_seed is None
    batch.reqs = [batch.reqs[2], batch.reqs[1], batch.reqs[0]]
    with seeded_batch(batch):
        assert torch.equal(batch.sampling_info.sampling_seed, first.flip(0))
    assert batch.sampling_info.sampling_seed is None


def test_unseeded_requests_do_not_enable_a_seeded_path():
    batch = _batch([None, None])
    with seeded_batch(batch):
        assert batch.sampling_info.sampling_seed is None
    assert not hasattr(batch.reqs[0], "_dspine_random_seed")


def test_prefill_seed_repeats_with_filtering_and_sampling_mask_capture():
    info = sampling_info(2, temperature=1.0, top_k=2, top_p=0.95)
    info.sampling_seed = torch.tensor([2, -3])
    info.sampling_mask_batch_indices = torch.tensor([1])
    probs = torch.tensor([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]])
    positions = torch.tensor([11, 29])
    with patch(
        "sglang.srt.runtime_context.get_exec",
        return_value=SimpleNamespace(
            kernel=SimpleNamespace(sampling_backend="pytorch")
        ),
    ):
        first, capture = sample_prefill(probs, info, positions)
        torch.rand(17)
        second, _ = sample_prefill(probs, info, positions)
    assert torch.equal(first, second)
    assert capture.batch_rows.tolist() == [1]
    assert capture.weights[0, 2] == 0
    assert capture.selected_weight[0] == capture.weights[0, first[1]]


def test_prefill_samples_the_requested_target_distribution():
    n = 8192
    info = sampling_info(n, temperature=1.0)
    info.sampling_seed = torch.arange(n)
    probs = torch.tensor([0.2, 0.3, 0.5]).expand(n, 3)
    with patch(
        "sglang.srt.runtime_context.get_exec",
        return_value=SimpleNamespace(
            kernel=SimpleNamespace(sampling_backend="pytorch")
        ),
    ):
        tokens, _ = sample_prefill(probs, info, torch.zeros(n, dtype=torch.int64))
    frequencies = torch.bincount(tokens.long(), minlength=3).float() / n
    assert torch.max(torch.abs(frequencies - probs[0])) < 0.02


def test_target_distribution_preserves_request_seeds_and_rng():
    from sglang_dspine.sampling.target import TargetDistribution

    info = sampling_info(17, temperature=0.7, top_k=50, top_p=0.8)
    info.sampling_seed = torch.arange(17) + 7
    seeds = info.sampling_seed.clone()
    logits = torch.zeros(17 * 16, 3)
    expected = torch.full((17, 16, 3), 1 / 3)
    distribution = TargetDistribution(SimpleNamespace(block_size=16))
    rng = torch.get_rng_state().clone()
    with (
        patch("sglang_dspine.runtime.worker.build_dspine_target_probs", return_value=expected) as build,
        patch("sglang.srt.runtime_context.get_exec", return_value=SimpleNamespace(
            kernel=SimpleNamespace(sampling_backend="pytorch"))),
    ):
        assert distribution.dense(logits, info, 17) is expected
    assert build.call_args.kwargs["sampling_info"] is info
    assert torch.equal(info.sampling_seed, seeds)
    assert torch.equal(torch.get_rng_state(), rng)


def test_prefill_hook_uses_seeded_sampler_and_preserves_unseeded_dispatch():
    from sglang_dspine.plugin import around_seeded_sampler

    info = SimpleNamespace(sampling_seed=torch.tensor([7]))
    sampler, probs, positions = object(), torch.ones(1, 3), torch.tensor([4])
    with (
        patch(
            "sglang.srt.runtime_context.get_spec",
            return_value=SimpleNamespace(speculative_algorithm="DSPINE"),
        ),
        patch(
            "sglang_dspine.sampling.random.sample_prefill",
            return_value=("seeded", None),
        ) as seeded,
    ):
        assert around_seeded_sampler(None, sampler, probs, info, positions, False) == (
            "seeded",
            None,
        )
        seeded.assert_called_once_with(probs, info, positions)
        info.sampling_seed = None
        original = lambda *args: args
        assert around_seeded_sampler(
            original, sampler, probs, info, positions, False
        ) == (sampler, probs, info, positions, False)


def test_worker_scope_and_registered_prefill_hook_honor_api_seed():
    from sglang.srt.layers.sampler import Sampler
    from sglang.srt.plugins.hook_registry import HookRegistry
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    from sglang_dspine import plugin
    from sglang_dspine.runtime.worker import DSpineWorkerV2

    plugin.register()
    HookRegistry.apply_hooks()
    worker = object.__new__(DSpineWorkerV2)
    worker.verify_caps = None
    worker._dspine_sampler = SimpleNamespace(optimized=False)
    sampler = object.__new__(Sampler)
    batch = _batch([7, -19])
    batch.sampling_info = sampling_info(2, temperature=1.0, top_k=20, top_p=0.95)
    batch.sampling_info.sampling_seed = None
    probabilities = torch.tensor([[0.2, 0.3, 0.5], [0.5, 0.4, 0.1]])
    positions = torch.tensor([11, 199])

    def target_prefill(parent, batch):
        assert batch.sampling_info.sampling_seed.tolist() == [7, -19]
        return sampler._sample_from_probs(
            probabilities, batch.sampling_info, positions, False
        )

    with (
        patch.object(DFlashWorkerV2, "forward_batch_generation", target_prefill),
        patch(
            "sglang.srt.runtime_context.get_spec",
            return_value=SimpleNamespace(speculative_algorithm="DSPINE"),
        ),
        patch(
            "sglang.srt.runtime_context.get_exec",
            return_value=SimpleNamespace(
                kernel=SimpleNamespace(sampling_backend="pytorch")
            ),
        ),
    ):
        first, _ = worker.forward_batch_generation(batch=batch)
        assert batch.sampling_info.sampling_seed is None
        torch.rand(257)
        second, _ = worker.forward_batch_generation(batch=batch)
        assert torch.equal(first, second)
        assert batch.sampling_info.sampling_seed is None


@pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1", reason="Triton interpreter run"
)
@pytest.mark.parametrize("stream", [DRAFT, ACCEPT, BONUS, PREFILL])
def test_triton_draws_match_cpu_bitwise(stream):
    seeds = torch.tensor([0, -1, -(2**63), 2**63 - 1, 2**40 + 7])
    positions = torch.tensor([0, 19, 214, 8192, 131071])
    actual = torch.empty(5, 15)
    _uniforms[(5,)](seeds, positions, actual, 15, stream, 16)
    assert torch.equal(actual, seeded_uniforms(seeds, positions, 15, stream))
