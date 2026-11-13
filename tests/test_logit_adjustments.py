"""Verification transforms agree with native AR on actual signed logits."""

from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

from sglang_dspine.sampling.compact import (
    adjust_logits,
    around_verify_logits_adjustments,
    can_keep_compact,
)


def _native_info(bs, vocab, device):
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

    info = object.__new__(SamplingBatchInfo)
    info.temperatures = torch.ones(bs, 1, device=device)
    info.vocab_size = vocab
    info.device = device
    return info


def _live_penalizers(bs, vocab, device, *, repetition_first=False):
    """Real pinned penalty classes with populated decode state; no mocked apply."""
    from sglang.srt.sampling.penaltylib.frequency_penalty import BatchedFrequencyPenalizer
    from sglang.srt.sampling.penaltylib.min_new_tokens import BatchedMinNewTokensPenalizer
    from sglang.srt.sampling.penaltylib.orchestrator import BatchedPenalizerOrchestrator
    from sglang.srt.sampling.penaltylib.presence_penalty import BatchedPresencePenalizer
    from sglang.srt.sampling.penaltylib.repetition_penalty import BatchedRepetitionPenalizer

    orch = object.__new__(BatchedPenalizerOrchestrator)
    orch.is_required = True
    orch.vocab_size, orch.device = vocab, device
    frequency = BatchedFrequencyPenalizer(orch)
    frequency.cumulated_frequency_penalties = (
        torch.arange(bs * vocab, device=device).view(bs, vocab).float() % 5 / 3
    )
    presence = BatchedPresencePenalizer(orch)
    presence.cumulated_presence_penalties = (
        torch.arange(bs * vocab, device=device).view(bs, vocab).float() % 2 * .7
    )
    repetition = BatchedRepetitionPenalizer(orch)
    repetition.cumulated_repetition_penalties = (
        0.5 + torch.arange(bs * vocab, device=device).view(bs, vocab).float() % 4
    )
    minimum = BatchedMinNewTokensPenalizer(orch)
    minimum.min_new_tokens = torch.full((bs, 1), 3, device=device)
    minimum.len_output_tokens = torch.arange(bs, device=device)[:, None]
    minimum.stop_token_penalties = torch.zeros(bs, vocab, device=device)
    minimum.stop_token_penalties[:, -1] = -torch.inf
    ordered = [frequency, presence, repetition, minimum]
    if repetition_first:
        ordered = [repetition, frequency, presence, minimum]
    for penalty in ordered:
        penalty._is_prepared = True
    orch.penalizers = OrderedDict((type(p), p) for p in ordered)
    return orch


def _reference(info, logits, width):
    # AR's native transform is independent of the DFlash verification helper.
    result = logits.view(len(info), width, -1).clone()
    for depth in range(width):
        info.apply_logits_bias(result[:, depth])
    return result.flatten(0, 1)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bs", [1, 3, 9])
@pytest.mark.parametrize("mode", ["repetition", "overlap", "live", "live_reversed"])
def test_overlap_and_live_penalties_match_native_signed_logits(device, dtype, bs, mode):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    width, vocab = 5, 11
    info = _native_info(bs, vocab, device)
    orch = _live_penalizers(bs, vocab, device, repetition_first=mode == "live_reversed")
    if mode.startswith("live"):
        info.penalizer_orchestrator = orch
    else:
        info.acc_scaling_penalties = orch.accumulate_scaling_penalties()
        if mode == "overlap":
            info.acc_additive_penalties = torch.zeros(bs, vocab, device=device)
            orch.accumulate_additive_penalties(info.acc_additive_penalties)
    info.logit_bias = (
        torch.arange(bs * vocab, device=device).view(bs, vocab).float() % 3 - 1
    ) * .23
    logits = torch.linspace(-4.0, 4.0, bs * width * vocab, device=device).view(-1, vocab).to(dtype)
    expected = _reference(info, logits, width)
    # Nonuniform, noncontiguous request ownership also covers ragged packing.
    rows = torch.arange(bs * width, device=device)
    selected = rows[(rows % width) <= (rows // width) % width].flip(0)
    owners = selected // width
    packed = logits.index_select(0, selected)
    adjust_logits(packed, info, owners)
    assert torch.equal(packed, expected.index_select(0, selected))
    full = logits.clone()
    adjust_logits(full.view(bs, width, vocab), info, None)
    assert torch.equal(full, expected)


def test_linear_alias_is_supported_without_adding_both_names():
    info = SimpleNamespace(acc_linear_penalties=torch.tensor([[1.0, -1.0]]),
                           acc_additive_penalties=torch.tensor([[.5, -.5]]))
    original = torch.tensor([[2.0, -2.0]])
    current = original.clone()
    adjust_logits(current, info, torch.tensor([0]))
    assert torch.equal(current, torch.tensor([[2.5, -2.5]]))
    info.acc_additive_penalties = None
    legacy = original.clone()
    adjust_logits(legacy, info, torch.tensor([0]))
    assert torch.equal(legacy, torch.tensor([[3.0, -3.0]]))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_no_adjustments_leave_all_bits_and_rng_unchanged(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    logits = torch.tensor([[float("nan"), -0.0, 0.0, -3.0, float("inf")]], device=device)
    before = logits.view(torch.int32).clone()
    rng = torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()
    for info in (None, SimpleNamespace()):
        adjust_logits(logits, info, torch.tensor([0], device=device))
    assert torch.equal(logits.view(torch.int32), before)
    current_rng = torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()
    assert torch.equal(rng, current_rng)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_grammar_is_after_penalties_and_before_logit_bias(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from sglang.srt.constrained.base_grammar_backend import GrammarMask

    class BooleanGrammar:
        def apply_vocab_mask(self, *, logits, vocab_mask):
            logits.masked_fill_(~vocab_mask, -torch.inf)

    info = _native_info(2, 3, device)
    info.acc_additive_penalties = torch.tensor([[1., -3., .5], [2., 1., -1.]], device=device)
    info.acc_scaling_penalties = torch.full((2, 3), 2., device=device)
    info.logit_bias = torch.tensor([[.2, .3, .4], [.1, .2, .3]], device=device)
    info.grammar_mask = GrammarMask(BooleanGrammar(), torch.tensor(
        [[True, False, True], [False, True, True]], device=device))
    original = torch.tensor([[2., -2., 1.], [-2., 2., -1.]] * 2, device=device)
    expected = _reference(info, original, 2)
    selected = torch.tensor([0, 2, 3], device=device)
    actual = original[selected].clone()
    adjust_logits(actual, info, selected // 2)
    assert torch.equal(actual, expected[selected])
    full = original.clone()
    adjust_logits(full.view(2, 2, 3), info, None)
    assert torch.equal(full, expected)


def test_full_envelope_runs_native_custom_processor_before_penalties(monkeypatch):
    from sglang.srt.layers.sampler import apply_custom_logit_processor
    from sglang.srt.sampling.sampling_batch_info import ProcessorEntry
    from sglang_dspine.verification import caps

    info = _native_info(2, 3, "cpu")
    info.has_custom_logit_processor = True
    info.custom_params = [{"offset": 1.0}, {"offset": -1.0}]
    info.custom_logit_processor = {0: ProcessorEntry(
        processor=lambda logits, params: logits * 2 + torch.tensor(
            [p["offset"] for p in params])[:, None],
        rows=[0, 1], indices=torch.tensor([0, 1]))}
    info.acc_scaling_penalties = torch.full((2, 3), 2.)
    logits = torch.tensor([[2., -2., .5], [-2., 2., -.5]] * 2)
    expected = logits.clone()
    apply_custom_logit_processor(expected, info, num_tokens_in_batch=2)
    expected = _reference(info, expected, 2)
    monkeypatch.setattr(caps, "active_controller", lambda: SimpleNamespace(compact_round=False))
    def broken_native(**kwargs):
        raise AssertionError("DSpine must not call the zero-bias native verification helper")
    actual = logits.clone()
    around_verify_logits_adjustments(broken_native, next_token_logits=actual,
                                    sampling_info=info, draft_token_num=2)
    assert torch.equal(actual, expected)
    assert not can_keep_compact(info)


def test_generic_dspine_without_controller_keeps_penalties(monkeypatch):
    from sglang_dspine.runtime import worker
    from sglang_dspine.verification import caps

    monkeypatch.setattr(caps, "active_controller", lambda: None)
    monkeypatch.setattr(worker, "_active_worker", SimpleNamespace())
    info = _native_info(1, 2, "cpu")
    info.acc_scaling_penalties = torch.full((1, 2), 2.)
    logits = torch.tensor([[2., -2.], [-2., 2.]])
    expected = _reference(info, logits, 2)

    def broken_native(**kwargs):
        raise AssertionError("Generic DSpine must also preserve repetition penalties")

    around_verify_logits_adjustments(
        broken_native, next_token_logits=logits,
        sampling_info=info, draft_token_num=2,
    )
    assert torch.equal(logits, expected)


def test_native_dflash_without_dspine_keeps_original_adjustments(monkeypatch):
    from sglang_dspine.runtime import worker
    from sglang_dspine.verification import caps

    monkeypatch.setattr(caps, "active_controller", lambda: None)
    monkeypatch.setattr(worker, "_active_worker", None)
    logits = torch.tensor([[2., -2.]])
    info, result = object(), object()

    def original(**kwargs):
        assert kwargs["next_token_logits"] is logits
        assert kwargs["sampling_info"] is info
        assert kwargs["draft_token_num"] == 1
        return result

    assert around_verify_logits_adjustments(
        original, next_token_logits=logits, sampling_info=info, draft_token_num=1,
    ) is result


@pytest.mark.parametrize("mode", ["overlap", "live"])
def test_compact_penalties_capture_without_host_synchronization(mode):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    bs, vocab = 3, 11
    info = _native_info(bs, vocab, "cuda")
    orch = _live_penalizers(bs, vocab, "cuda")
    if mode == "live":
        info.penalizer_orchestrator = orch
    else:
        info.acc_scaling_penalties = orch.accumulate_scaling_penalties()
        info.acc_additive_penalties = torch.zeros(bs, vocab, device="cuda")
        orch.accumulate_additive_penalties(info.acc_additive_penalties)
    owners = torch.tensor([0, 1, 1, 2, 2, 2], device="cuda")
    initial = torch.linspace(-3., 3., owners.numel() * vocab, device="cuda").view(-1, vocab)
    expected = initial.clone()
    adjust_logits(expected, info, owners)
    actual = initial.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual.copy_(initial)
        adjust_logits(actual, info, owners)
    graph.replay()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
