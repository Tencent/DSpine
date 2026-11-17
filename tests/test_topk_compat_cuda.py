# Copyright 2026 Tencent
# Licensed under the Apache License, Version 2.0.

"""Real CUDA numerical and graph checks for unsupported filtered Top-K devices."""

import pytest
import torch

from sglang_dspine.layers.fp8 import coarse

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _forbidden_flashinfer(*args, **kwargs):
    raise AssertionError("FlashInfer must not run on the portable path")


def _cpu_lexical_ids(scores, k, token_ids=None):
    assert scores.device.type == "cpu"
    rows = scores.float().tolist()
    ids = (
        [list(range(scores.shape[1])) for _ in rows]
        if token_ids is None
        else token_ids.tolist()
    )
    return torch.tensor(
        [
            [token for _, token in sorted(zip(row, row_ids), key=lambda x: (-x[0], x[1]))[:k]]
            for row, row_ids in zip(rows, ids)
        ],
        dtype=torch.int64,
    )


def _large_scores(dtype, k, phase):
    scores = torch.randn(3, 131072, generator=torch.Generator().manual_seed(71 + phase))
    scores = scores.to(dtype)
    tie_start = 17 + 211 * phase
    scores[1].zero_()
    scores[1, tie_start : tie_start + k + 32] = 3
    scores[2, -3:] = -torch.inf
    scores[2, -4] = 16
    expected_ids = _cpu_lexical_ids(scores, k)
    assert expected_ids[1].tolist() == list(range(tie_start, tie_start + k))
    assert expected_ids[2, 0].item() == scores.shape[1] - 4
    assert torch.all(expected_ids[2] < scores.shape[1] - 3)
    return scores, expected_ids


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("k", [16, 128])
def test_portable_topk_eager_and_graph(monkeypatch, dtype, k):
    monkeypatch.setattr(coarse, "supports_filtered_topk", lambda device: False)
    cpu_scores, expected_ids = _large_scores(dtype, k, 0)
    scores = cpu_scores.cuda()
    values, ids = coarse.stable_local_topk(scores, k, _forbidden_flashinfer)
    torch.testing.assert_close(ids.cpu(), expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(
        values.cpu(), cpu_scores.gather(-1, expected_ids), rtol=0, atol=0
    )
    for _ in range(3):
        coarse.stable_local_topk(scores, k, _forbidden_flashinfer)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            replay_values, replay_ids = coarse.stable_local_topk(
                scores, k, _forbidden_flashinfer
            )
        for phase in (1, 2):
            cpu_scores, expected_ids = _large_scores(dtype, k, phase)
            scores.copy_(cpu_scores)
            graph.replay()
            torch.testing.assert_close(replay_ids.cpu(), expected_ids, rtol=0, atol=0)
            torch.testing.assert_close(
                replay_values.cpu(), cpu_scores.gather(-1, expected_ids), rtol=0, atol=0
            )
    finally:
        graph.reset()


def _coarse_reference(local, hidden, cpu_weight):
    scale = hidden.float().abs().amax(-1, keepdim=True) / 448
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quantized = (hidden.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    coarse_scores = torch.ops.aten._scaled_mm.default(
        quantized,
        local.weight8_t,
        scale,
        local.weight_scale,
        out_dtype=torch.bfloat16,
        use_fast_accum=False,
    )
    coarse_ids = _cpu_lexical_ids(coarse_scores.cpu(), 128)
    cpu_scores = (
        cpu_weight[coarse_ids].float() * hidden.cpu().float()[:, None]
    ).sum(-1).bfloat16()
    final_ids = _cpu_lexical_ids(cpu_scores, 16, coarse_ids).sort(-1).values
    return final_ids, coarse_ids, cpu_scores


def test_portable_full_vocab_coarse_eager_and_graph(monkeypatch):
    from sglang_dspine.layers.fp8 import gather

    monkeypatch.setattr(coarse, "supports_filtered_topk", lambda device: False)
    # Exact two-term BF16 dots keep the final rescore reference independent of kernels.
    cpu_weight = torch.zeros(512, 512, dtype=torch.bfloat16)
    cpu_weight[:, 0] = torch.arange(512).remainder(47).bfloat16() / 32
    cpu_weight[:, 1] = torch.arange(512).remainder(31).bfloat16() / 64
    local = coarse.FullVFP8Coarse(torch, cpu_weight.cuda())
    hidden = torch.zeros(4, 512, device="cuda", dtype=torch.bfloat16)
    hidden[:, :2] = torch.tensor([[1, 2], [2, -1], [-1, 1], [0, 0]], device="cuda")

    def retrieve():
        return local.retrieve_and_rescore(
            hidden, fi_top_k=_forbidden_flashinfer, gather_ops=gather
        )

    def check(actual):
        expected = _coarse_reference(local, hidden, cpu_weight)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result.cpu(), reference, rtol=0, atol=0)

    check(retrieve())
    for _ in range(3):
        retrieve()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            replayed = retrieve()
        for _ in range(2):
            hidden.neg_()
            graph.replay()
            check(replayed)
    finally:
        graph.reset()
