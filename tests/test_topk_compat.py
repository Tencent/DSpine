# Copyright 2026 Tencent
# Licensed under the Apache License, Version 2.0.

"""Capability routing checks that do not require FlashInfer or a CUDA device."""

import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from sglang_dspine.layers.fp8 import coarse


def _forbidden_flashinfer(*args, **kwargs):
    raise AssertionError("FlashInfer must not run on the portable path")


@pytest.fixture
def capability_cache():
    coarse.supports_filtered_topk.cache_clear()
    yield
    coarse.supports_filtered_topk.cache_clear()


def test_cpu_capability_does_not_query_flashinfer(monkeypatch, capability_cache):
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.topk",
        SimpleNamespace(can_implement_filtered_topk=_forbidden_flashinfer),
    )
    monkeypatch.setattr(torch.cuda, "device", _forbidden_flashinfer)
    assert coarse.supports_filtered_topk(torch.device("cpu")) is False


def test_capability_is_cached_in_each_explicit_device_context(
    monkeypatch, capability_cache
):
    active_devices = []
    queried_devices = []

    @contextmanager
    def device_context(device):
        active_devices.append(device.index)
        try:
            yield
        finally:
            active_devices.pop()

    def can_implement_filtered_topk():
        assert len(active_devices) == 1
        index = active_devices[-1]
        queried_devices.append(index)
        return index == 1

    monkeypatch.setattr(torch.cuda, "device", device_context)
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.topk",
        SimpleNamespace(can_implement_filtered_topk=can_implement_filtered_topk),
    )
    for _ in range(2):
        assert coarse.supports_filtered_topk(torch.device("cuda:0")) is False
        assert coarse.supports_filtered_topk(torch.device("cuda:1")) is True
    assert queried_devices == [0, 1]
    assert active_devices == []


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("k", [16, 128])
def test_cpu_topk_keeps_lower_ids_at_ties(dtype, k):
    scores = torch.arange(257).remainder(7).to(dtype).repeat(2, 1)
    scores[1, 254:] = -torch.inf
    expected_ids = torch.tensor(
        [
            sorted(range(scores.shape[1]), key=lambda i: (-row[i], i))[:k]
            for row in scores.tolist()
        ]
    )
    values, ids = coarse.stable_local_topk(scores, k, _forbidden_flashinfer)
    assert ids.is_contiguous(), "Pointer-based gather kernels require packed rows"
    torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(values, scores.gather(-1, expected_ids), rtol=0, atol=0)


def test_supported_topk_preserves_flashinfer_arguments(monkeypatch):
    scores = SimpleNamespace(device=torch.device("cuda:1"))
    expected_values = torch.tensor([[5.0, 4.0]])
    expected_ids = torch.tensor([[2, 1]], dtype=torch.int32)
    observed = []

    def supported(device):
        assert device == scores.device
        return True

    def top_k(actual_scores, k, **kwargs):
        assert actual_scores is scores
        observed.append((k, kwargs))
        return expected_values, expected_ids

    monkeypatch.setattr(coarse, "supports_filtered_topk", supported)
    values, ids = coarse.stable_local_topk(scores, 2, top_k)
    assert values is expected_values
    assert ids.dtype == torch.int64
    torch.testing.assert_close(ids, expected_ids.long(), rtol=0, atol=0)
    assert observed == [
        (
            2,
            {
                "sorted": False,
                "deterministic": True,
                "tie_break": 1,
                "dsa_graph_safe": True,
            },
        )
    ]
