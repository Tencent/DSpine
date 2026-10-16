"""CPU tests for worker and graph-runner integration with DFlashWorkerV2.
Stand-ins verify proposal routing and sampled execution after graph replay."""

import logging
import math
import types
from unittest import mock

import pytest
import torch
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)

from sglang_dspine.graph_runner import DSpineDraftCudaGraphRunner
from sglang_dspine.models.dspine import DSpineCodeBridge
from sglang_dspine.worker import DSpineDraftSampler, DSpineWorkerV2

from ref_utils import sampling_info


def _graph_runner(sampler):
    runner = object.__new__(DSpineDraftCudaGraphRunner)
    runner.model_runner = types.SimpleNamespace(
        model=types.SimpleNamespace(dspine_sampler=sampler)
    )
    return runner


def test_execute_replays_then_hands_the_batch_to_the_sampler():
    events = []
    output, batch = object(), types.SimpleNamespace(batch_size=2)

    def base_execute(self, forward_batch, pp_proxy_tensors=None):
        events.append(("replay", forward_batch, pp_proxy_tensors))
        return output

    sampler = types.SimpleNamespace(
        after_graph_replay=lambda forward_batch: events.append(
            ("sample", forward_batch)
        )
    )
    with mock.patch.object(DecodeCudaGraphRunner, "execute", base_execute):
        assert _graph_runner(sampler).execute(batch, pp_proxy_tensors="pp") is output
        assert events == [("replay", batch, "pp"), ("sample", batch)]
        with pytest.raises(RuntimeError):
            _graph_runner(None).execute(batch)


def _cpu_sampler(T=4, H=512, C=64, R=128, L=2, V=300, k=8, max_bs=4):
    cfg = types.SimpleNamespace(
        hidden_size=H,
        num_hidden_layers=L,
        block_size=T,
        vocab_size=V,
        dflash_config={
            "selected_prefix_terminal": False,
            "selected_prefix_feature_write_mode": "independent",
            "selected_prefix_feature_writes": True,
            "draft_block_causal": True,
            "selected_prefix_feature_layers": list(range(1, L + 1)),
            "selected_prefix_code_bridge": {"code_size": C, "rank": R, "eps": 1e-6},
        },
    )
    bridge = DSpineCodeBridge(cfg).to(torch.bfloat16)
    g = torch.Generator().manual_seed(0)
    for name, t in list(bridge.named_parameters()) + list(bridge.named_buffers()):
        if name == "packed_weight":
            continue
        if name.endswith("_version"):
            value = torch.ones((), dtype=torch.int64)
        elif name == "out.weight":
            value = torch.zeros(t.shape)
        else:
            value = torch.randn(t.shape, generator=g) / math.sqrt(t.shape[-1])
        bridge.load_weight(name, value.to(t.dtype))
    bridge.prepare()
    draft_model = types.SimpleNamespace(
        selected_prefix_bridge=bridge,
        head_topk=k,
        layers=[None] * L,
        norm=types.SimpleNamespace(
            weight=torch.ones(H, dtype=torch.bfloat16), variance_epsilon=1e-6
        ),
        last_write_stash=None,
    )
    lm = (torch.randn(V, H, generator=g) / math.sqrt(H)).to(torch.bfloat16)
    return DSpineDraftSampler(
        draft_model=draft_model,
        lm_head_weight=lm,
        block_size=T,
        max_bs=max_bs,
        device="cpu",
    )


def test_sampled_step_runs_after_a_replay_only_in_a_round_with_a_sampled_request():
    sampler = _cpu_sampler()
    drawn = []

    def fake_sample(bs):  # the real step is covered by the CUDA tests
        drawn.append(bs)
        sampler._round_sampled = True

    sampler.sample = fake_sample
    replay = types.SimpleNamespace(batch_size=3)
    # all-greedy round: the replay's greedy chain is the proposal
    sampler.stage_sampling_params(bs=3, sampling_info=None)
    sampler.after_graph_replay(replay)
    assert drawn == [] and not sampler.round_sampled
    # a round with a sampled request samples after every replay of it
    info = sampling_info(3, temperature=0.5, top_k=20, top_p=0.9, min_p=0.05)
    info.top_ks[1] = 1  # a greedy request in the same batch
    sampler.stage_sampling_params(bs=3, sampling_info=info)
    assert not sampler.round_sampled
    sampler.after_graph_replay(replay)
    assert drawn == [3] and sampler.round_sampled
    sampler.after_graph_replay(replay)
    assert drawn == [3, 3]
    # staged per request: 1 / 0.5 = 2; top_k 20 caps at K = 8; top_k 1 marks the greedy row
    assert sampler.inv_temp[:3].tolist() == [2.0, 2.0, 2.0]
    assert sampler.top_k[:3].tolist() == [8, 1, 8]
    assert sampler.greedy[:3].tolist() == [0, 1, 0]
    # a replay of another batch than the staged one is refused
    with pytest.raises(RuntimeError):
        sampler.after_graph_replay(types.SimpleNamespace(batch_size=2))
    # the next round starts unsampled
    sampler.stage_sampling_params(bs=3, sampling_info=info)
    assert not sampler.round_sampled


class _RecordingSampler:
    """Stand-in for DSpineDraftSampler that records what the worker asks of it."""

    def __init__(self, bs, gamma, k):
        self.calls = []
        self.out = torch.arange(bs * gamma, dtype=torch.int64)
        self.candidate_out = torch.zeros((bs, gamma, k), dtype=torch.int64)
        self.q_out = torch.zeros((bs, gamma, k))
        self.round_sampled = False

    def stage_sampling_params(self, *, bs, sampling_info):
        self.calls.append(("stage", bs, sampling_info))

    def __call__(self, hidden_states, input_ids):
        self.calls.append(("tail", tuple(hidden_states.shape), input_ids.clone()))

    def sample(self, bs):
        self.calls.append(("sample", bs))
        self.round_sampled = True


def _worker(sampler, block_size):
    worker = object.__new__(DSpineWorkerV2)
    worker._dspine_sampler = sampler
    worker.block_size = block_size
    worker._selector_sample = None
    return worker


def test_eager_path_stages_runs_the_tail_and_samples_only_a_sampled_round():
    bs, T, H, k = 2, 4, 8, 8
    sampler = _RecordingSampler(bs, T - 1, k)
    worker = _worker(sampler, T)
    draft = types.SimpleNamespace(hidden_states=torch.zeros(bs * T, H))
    anchors = torch.tensor([7, 9])

    proposal = worker._propose_selector_block(
        draft_logits_output=draft,
        bs=bs,
        lm_head=None,
        anchor_token_ids=anchors,
        sampling_info=None,
    )
    assert [c[0] for c in sampler.calls] == ["stage", "tail"]
    block_ids = sampler.calls[1][2].view(bs, T)
    assert block_ids[:, 0].tolist() == [7, 9]  # the tail reads only the anchors
    assert worker._selector_sample is None
    assert torch.equal(proposal, sampler.out.view(bs, T - 1))

    sampler.calls.clear()
    info = sampling_info(bs, temperature=1.0)
    proposal = worker._propose_selector_block(
        draft_logits_output=draft,
        bs=bs,
        lm_head=None,
        anchor_token_ids=anchors,
        sampling_info=info,
    )
    assert [c[0] for c in sampler.calls] == ["stage", "tail", "sample"]
    assert sampler.calls[0][1:] == (bs, info) and sampler.calls[2][1] == bs
    candidate_ids, q_rows = worker._selector_sample
    assert candidate_ids.data_ptr() == sampler.candidate_out.data_ptr()
    assert q_rows.data_ptr() == sampler.q_out.data_ptr()
    assert torch.equal(proposal, sampler.out.view(bs, T - 1))


def test_seed_warning_reads_the_rank_from_parallel_state(caplog):
    # SGLang 0.5.20: ModelRunner has no tp_rank; the worker reads its rank from ps.
    worker = _worker(None, 4)
    worker.ps = types.SimpleNamespace(tp_rank=0)
    worker.model_runner = types.SimpleNamespace()
    worker._warned_sampling_seed = False
    info = types.SimpleNamespace(is_all_greedy=False, sampling_seed=torch.tensor([7]))
    batch = types.SimpleNamespace(sampling_info=info)
    with caplog.at_level(logging.WARNING, logger="sglang_dspine.worker"):
        worker._validate_phase1_sampling_support(batch)
        worker._validate_phase1_sampling_support(batch)
    assert worker._warned_sampling_seed
    warned = [r for r in caplog.records if "sampling_seed is not" in r.getMessage()]
    assert len(warned) == 1


def test_sampled_verify_refuses_a_round_whose_sampled_chain_did_not_run():
    worker = _worker(types.SimpleNamespace(round_sampled=False), 4)
    with pytest.raises(RuntimeError, match="sampled draft chain"):
        worker._selector_sampling_accept(
            candidates=torch.zeros((1, 4), dtype=torch.int64),
            next_token_logits=torch.zeros((4, 16)),
            candidate_ids=torch.zeros((1, 3, 8), dtype=torch.int64),
            q_rows=torch.full((1, 3, 8), 1 / 8),
            sampling_info=sampling_info(1, temperature=1.0),
            draft_input=None,
        )
