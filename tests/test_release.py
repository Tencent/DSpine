"""Padding eligibility and portable target-distribution fallback."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang_dspine.sampling.padding import logits_padding
from sglang_dspine.sampling.target import TargetDistribution


def _controller(**changes):
    params = dict(
        temperature=1.0,
        top_k=20,
        top_p=0.95,
        min_p=0.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        min_new_tokens=0,
    )
    params.update(changes)
    info = SimpleNamespace(
        is_all_greedy=False,
        is_any_greedy=False,
        need_top_k_sampling=True,
        need_top_p_sampling=True,
        need_min_p_sampling=False,
    )
    batch = SimpleNamespace(
        has_grammar=False,
        return_logprob=False,
        sampling_info=info,
        reqs=[SimpleNamespace(sampling_params=SimpleNamespace(**params))],
    )
    return SimpleNamespace(
        sampled_round=True,
        worker=SimpleNamespace(sampled_policy="sampler"),
        saved_input=(None, None, batch),
    )


def test_padding_preserves_verified_logits_and_normalizes_unused_rows():
    verified = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    actual = logits_padding(_controller(), verified, (4, 3))
    actual[:2].copy_(verified)
    assert torch.equal(actual[:2], verified)
    assert torch.equal(
        actual[2:].softmax(-1), torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    )


@pytest.mark.parametrize(
    "changes",
    [
        dict(temperature=0.7),
        dict(top_k=10),
        dict(top_p=0.9),
        dict(presence_penalty=1.0),
        dict(regex="a+"),
    ],
)
def test_neutral_padding_is_independent_of_request_sampling_parameters(changes):
    actual = logits_padding(_controller(**changes), torch.empty(2, 3), (4, 3))
    assert torch.equal(actual.softmax(-1), torch.tensor([[1.0, 0.0, 0.0]]).expand(4, 3))


def test_target_initialization_needs_no_device_geometry():
    with patch.object(
        torch.cuda, "get_device_properties", side_effect=AssertionError("GPU query")
    ):
        distribution = TargetDistribution(SimpleNamespace(device="cuda"))
    assert distribution.scratch is None
