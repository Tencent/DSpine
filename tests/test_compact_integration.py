"""Packed verification preserves request-wise adjustments and target policies."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang_dspine.sampling.compact import adjust_logits, can_keep_compact
from sglang_dspine.sampling.target import TargetDistribution


class Info(SimpleNamespace):
    def __len__(self):
        return self.temperatures.shape[0]


def test_compact_adjustments_equal_gathering_adjusted_full_rows():
    requests, width, vocab = 3, 16, 19
    g = torch.Generator().manual_seed(471)
    logits = torch.randn(requests * width, vocab, generator=g)
    window = torch.tensor([0, 1, 16, 17, 18, 32, 33, 34, 35])
    owner = window // width
    info = Info(
        temperatures=torch.ones(requests, 1),
        acc_linear_penalties=torch.randn(requests, vocab, generator=g),
        logit_bias=torch.randn(requests, vocab, generator=g),
    )
    expected = logits.view(requests, width, vocab).clone()
    expected += info.acc_linear_penalties[:, None]
    expected += info.logit_bias[:, None]
    actual = logits[window].clone()
    adjust_logits(actual, info, owner)
    assert torch.equal(actual, expected.flatten(0, 1)[window])


def test_compact_repetition_uses_real_logits_before_logit_bias():
    owner = torch.tensor([0, 1, 1])
    info = Info(
        temperatures=torch.ones(2, 1),
        acc_scaling_penalties=torch.tensor([[2.0, 2.0], [.5, .5]]),
        logit_bias=torch.tensor([[.5, -.5], [1.0, -1.0]]),
    )
    actual = torch.tensor([[2.0, -2.0], [-3.0, 3.0], [4.0, -4.0]])
    adjust_logits(actual, info, owner)
    assert torch.equal(actual, torch.tensor([[1.5, -4.5], [-.5, 5.0], [9.0, -3.0]]))


@pytest.mark.parametrize("field", ["has_custom_logit_processor", "sampling_mask_batch_indices"])
def test_custom_processors_and_mask_capture_keep_full_envelope(field):
    info = SimpleNamespace(has_custom_logit_processor=False, sampling_mask_batch_indices=None)
    assert can_keep_compact(info)
    setattr(info, field, True if field.startswith("has_") else torch.tensor([0]))
    assert not can_keep_compact(info)


def test_compact_dense_fallback_maps_each_requests_sampling_parameters(monkeypatch):
    seen = {}
    def build(**kwargs):
        seen.update(kwargs)
        return kwargs["next_token_logits"].softmax(-1)
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", SimpleNamespace(
        get_exec=lambda: SimpleNamespace(kernel=SimpleNamespace(sampling_backend="flashinfer"))))
    monkeypatch.setitem(sys.modules, "sglang_dspine.runtime.worker", SimpleNamespace(
        build_dspine_target_probs=build))
    info = Info(temperatures=torch.tensor([[.3], [.8], [1.4]]),
                top_ks=torch.tensor([2, 7, 19]), top_ps=torch.tensor([.5, .8, .95]),
                min_ps=torch.tensor([.0, .1, .2]), is_any_greedy=False)
    owners = torch.tensor([0, 1, 1, 2, 2, 2])
    logits = torch.randn(6, 19)
    result = TargetDistribution(SimpleNamespace(block_size=16)).dense(logits, info, 3, owners)
    assert seen["block_size"] == 1 and seen["bs"] == 6
    assert seen["sampling_info"] is not info
    for field in ("temperatures", "top_ks", "top_ps", "min_ps"):
        assert torch.equal(getattr(seen["sampling_info"], field), getattr(info, field)[owners])
    assert result.shape == (6, 19)
