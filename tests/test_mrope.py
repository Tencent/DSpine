"""Packed MRoPE must not require an equal number of tokens per request."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang_dspine.verification.mrope import (
    around_spec_mrope_positions,
    ragged_mrope_positions,
)
from sglang_dspine.verification.ragged import ragged_scope


@pytest.mark.parametrize("lens", [[1, 14, 16], [1], [3, 7, 1, 12, 4]])
def test_text_positions_keep_the_exact_ragged_token_order(lens):
    positions = torch.arange(sum(lens)) + 100
    indptr = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32)
    result = ragged_mrope_positions(positions, indptr)
    assert result.shape == (3, sum(lens))
    assert torch.equal(result, torch.stack([positions] * 3))


def test_mixed_multimodal_positions_use_each_requests_delta_and_skip_ghost_rows():
    positions = torch.tensor([10, 11, 20, 30, 31, 32, 33, 0, 0])
    indptr = torch.tensor([0, 2, 3, 7], dtype=torch.int32)
    deltas = torch.tensor([[0], [15], [-3]])
    result = ragged_mrope_positions(positions, indptr, deltas)
    expected = torch.tensor([10, 11, 35, 27, 28, 29, 30, 0, 0])
    assert torch.equal(result, expected.expand(3, -1))


def test_uniform_layout_matches_native_batched_delta_broadcast():
    positions = torch.arange(48).view(3, 16)
    deltas = torch.tensor([[0], [11], [-4]])
    result = ragged_mrope_positions(
        positions, torch.tensor([0, 16, 32, 48], dtype=torch.int32), deltas
    )
    native = (positions + deltas).flatten().unsqueeze(0).repeat(3, 1)
    assert torch.equal(result, native)


def test_graph_padding_with_empty_request_slots_keeps_real_token_owners():
    positions = torch.tensor([10, 11, 20, 0])
    indptr = torch.tensor([0, 2, 2, 3, 3], dtype=torch.int32)
    deltas = torch.tensor([5, 999, -2, 999])
    result = ragged_mrope_positions(positions, indptr, deltas)
    assert torch.equal(result, torch.tensor([15, 16, 18, 0]).expand(3, -1))


def test_hook_only_changes_dspine_ragged_and_preserves_explicit_positions(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sglang_dspine.spec",
        SimpleNamespace(is_dspine=lambda algorithm: algorithm == "DSPINE"),
    )
    original_calls = []

    def original(*args):
        original_calls.append(args)
        return "native"

    positions = torch.arange(31)
    forward = SimpleNamespace()
    runner = SimpleNamespace(spec_algorithm="DSPINE", device="cpu")
    layout = SimpleNamespace(qo_indptr_device=torch.tensor([0, 1, 15, 31]))
    batch = SimpleNamespace(
        spec_info=SimpleNamespace(positions=positions, ragged_verify_layout=layout),
        multimodal_inputs=[None, None, None],
    )
    assert around_spec_mrope_positions(original, forward, runner, batch) == "native"
    with ragged_scope():
        around_spec_mrope_positions(original, forward, runner, batch)
        assert torch.equal(forward.mrope_positions, positions.expand(3, -1))
        around_spec_mrope_positions(original, forward, runner, batch, positions + 17)
        assert torch.equal(forward.mrope_positions, (positions + 17).expand(3, -1))
        runner.spec_algorithm = "DFLASH"
        assert around_spec_mrope_positions(original, forward, runner, batch) == "native"
        runner.spec_algorithm = "DSPINE"
        batch.spec_info.ragged_verify_layout = None
        assert around_spec_mrope_positions(original, forward, runner, batch) == "native"
    assert len(original_calls) == 3


def test_hook_applies_native_mixed_multimodal_deltas(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sglang_dspine.spec",
        SimpleNamespace(is_dspine=lambda algorithm: algorithm == "DSPINE"),
    )
    forward = SimpleNamespace()
    runner = SimpleNamespace(spec_algorithm="DSPINE", device="cpu")
    batch = SimpleNamespace(
        spec_info=SimpleNamespace(
            positions=torch.tensor([10, 11, 20, 30, 31]),
            ragged_verify_layout=SimpleNamespace(
                qo_indptr_device=torch.tensor([0, 2, 3, 5])
            ),
        ),
        multimodal_inputs=[
            None,
            SimpleNamespace(mrope_position_delta=torch.tensor([[5]])),
            SimpleNamespace(mrope_position_delta=torch.tensor([[-2]])),
        ],
    )
    with ragged_scope():
        around_spec_mrope_positions(
            lambda *a: pytest.fail("unexpected fallback"), forward, runner, batch
        )
    assert torch.equal(
        forward.mrope_positions, torch.tensor([10, 11, 25, 28, 29]).expand(3, -1)
    )


def test_mixed_ragged_positions_are_cuda_graph_safe():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    positions = torch.arange(31, device="cuda")
    indptr = torch.tensor([0, 1, 15, 31], device="cuda", dtype=torch.int32)
    deltas = torch.tensor([0, 11, -4], device="cuda")
    ragged_mrope_positions(positions, indptr, deltas)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = ragged_mrope_positions(positions, indptr, deltas)
    positions.add_(3)
    deltas.add_(2)
    graph.replay()
    expected = torch.cat([
        positions[:1] + deltas[0],
        positions[1:15] + deltas[1],
        positions[15:] + deltas[2],
    ])
    assert torch.equal(result, expected.expand(3, -1))
