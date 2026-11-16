"""Pinned DFlash acceptance reuse and mandatory DSpine fallback agreement."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from sglang.srt.speculative import dflash_worker_v2 as dflash
from sglang.srt.speculative.spec_tp_sync import SpecTpSync

from sglang_dspine.runtime.tp_sync import DSpineTpCoordinator
from sglang_dspine.runtime.worker import DSpineWorkerV2


class _Group:
    """Replay the leader's actual broadcast payloads on a divergent follower."""

    world_size = 2

    def __init__(self, rank, packets):
        self.rank_in_group = rank
        self.packets = packets
        self.calls = 0

    def broadcast(self, value, src):
        assert src == 0
        if self.rank_in_group == 0:
            self.packets.append(value.clone())
        else:
            value.copy_(self.packets[self.calls])
        self.calls += 1


def _worker(group):
    worker = object.__new__(DSpineWorkerV2)
    worker._target_worker = SimpleNamespace()
    worker.verify_caps = None
    worker._dspine_tp = DSpineTpCoordinator(group)
    worker._tp_sync = SpecTpSync(group)
    worker._selector_sample = None
    worker._use_triton_accept_bonus = False
    worker.block_size = 4
    worker.device = "cpu"
    worker._accept_bonus_buffer_cap = 0
    worker._accept_bonus_buffer_slot = 0
    return worker


def _inputs(rank, sampled=False):
    predictions = [[1, 2, 11, 12], [4, 5, 6, 13]] if rank == 0 else [[7] * 4] * 2
    return dict(
        candidates=torch.tensor([[8, 1, 2, 3], [9, 4, 5, 6]]),
        next_token_logits=torch.nn.functional.one_hot(
            torch.tensor(predictions).flatten(), 16
        ).float(),
        sampling_info=SimpleNamespace(is_all_greedy=not sampled),
        draft_input=SimpleNamespace(max_top_k=4, uniform_top_k_value=4),
        prefix_lens=torch.tensor([2**40, 2**40 + 32]),
        bs=2,
    )


@pytest.mark.parametrize("sync", ["all", "off", "rng"])
@pytest.mark.parametrize("mode", ["greedy", "sampled", "selector"])
def test_real_parent_accepts_once_and_reuses_native_sync(monkeypatch, sync, mode):
    sampled = mode != "greedy"
    monkeypatch.setenv("SGLANG_SPEC_TP_SYNC", sync)
    packets, results = [], []
    for rank in range(2):
        group = _Group(rank, packets)
        worker = _worker(group)
        inputs = _inputs(rank, sampled)
        # Keep the real DSpine actual-Q seam, replacing only its GPU verifier.
        worker._dspine_sampler = SimpleNamespace(
            round_sampled=True,
            round_bs=2,
            candidate_out=torch.arange(12).view(2, 3, 2),
            q_out=torch.tensor([0.25, 0.75]).expand(2, 3, 2).clone(),
        )
        if mode == "selector":
            worker._selector_sample = (
                worker._dspine_sampler.candidate_out,
                worker._dspine_sampler.q_out,
            )
        worker._dspine_verify = Mock(
            return_value=(
                torch.tensor([2, 3] if rank == 0 else [0, 0], dtype=torch.int32),
                torch.tensor([11, 13] if rank == 0 else [7, 7]),
            )
        )
        with (
            patch.object(dflash, "SIMULATE_ACC_LEN", 0),
            patch.object(
                dflash, "is_dflash_sampling_verify_available", return_value=True
            ),
            patch(
                "sglang_dspine.runtime.worker.is_dflash_sampling_verify_available",
                return_value=True,
            ),
            patch.object(
                dflash,
                "compute_dflash_sampling_correct_drafts_and_bonus",
                side_effect=worker._accept_sampled_round,
            ),
            patch.object(
                dflash, "_commit_accept", wraps=dflash._commit_accept
            ) as commit,
        ):
            result = worker._accept_block(**inputs)
        assert commit.call_count == 1
        # Parent eager-path sequence calculation stays deferred.
        assert result[4] is None
        assert result[0].dtype == result[1].dtype == torch.int32
        assert result[2].dtype == result[3].dtype == torch.int64
        if sampled:
            actual_q = worker._dspine_verify.call_args.kwargs
            assert (
                actual_q["q_rows"].data_ptr()
                == worker._dspine_sampler.q_out.data_ptr()
            )
            assert (
                actual_q["candidate_ids"].data_ptr()
                == worker._dspine_sampler.candidate_out.data_ptr()
            )
        native_enabled = sync == "all" or (sync == "rng" and sampled)
        assert group.calls == (2 if sampled and native_enabled else 1)
        assert bool(worker._dspine_tp._accept_packets) == (not native_enabled)
        results.append(result)
    for leader, follower in zip(*results):
        if leader is not None:
            torch.testing.assert_close(leader, follower, rtol=0, atol=0)
    assert results[1][0].tolist() == [2, 3]
    assert results[1][2].tolist() == [11, 13]


@pytest.mark.parametrize("sync", ["all", "off"])
@pytest.mark.parametrize("layout", ["capped", "ragged"])
def test_bypassed_accept_syncs_final_native_buffers_in_place(monkeypatch, sync, layout):
    monkeypatch.setenv("SGLANG_SPEC_TP_SYNC", sync)
    packets, results = [], []
    for rank in range(2):
        group = _Group(rank, packets)
        worker = _worker(group)
        controller = SimpleNamespace(
            in_round=True,
            active=3,
            full=4,
            ragged_round=layout == "ragged",
            observe=Mock(),
        )
        worker.verify_caps = controller
        rounds = []
        for step in range(2):
            inputs = _inputs(rank)
            result = worker._next_accept_bonus_buffers(2)
            accept, commit, bonus, output, new_seq = result
            accept.copy_(torch.tensor([1, 2] if rank == 0 else [0, 0]))
            commit.copy_(accept + 1)
            bonus.fill_(2**33 + step + rank)
            output.fill_(step + rank)
            new_seq.copy_(inputs["prefix_lens"] + commit)
            before = [(value.data_ptr(), value.dtype) for value in result]
            controller.accept = Mock(return_value=result)
            with patch.object(
                dflash, "_commit_accept", side_effect=AssertionError("recommit")
            ):
                actual = worker._accept_block(**inputs)
            assert actual is result
            assert before == [(value.data_ptr(), value.dtype) for value in actual]
            assert controller.observe.call_args.args[0] is accept
            assert accept.tolist() == [1, 2]
            assert new_seq.tolist() == [2**40 + 2, 2**40 + 35]
            rounds.append(tuple(value.clone() for value in actual))
            if step == 0:
                retained = tuple(value.clone() for value in actual[1:])
                previous = actual[1:]
            else:
                # Parent alternates durable output buffers; synchronization must too.
                for value, saved in zip(previous, retained):
                    torch.testing.assert_close(value, saved, rtol=0, atol=0)
        assert group.calls == 2
        assert len(worker._dspine_tp._accept_packets) == 1
        results.append(rounds)
    for leader_round, follower_round in zip(*results):
        for leader, follower in zip(leader_round, follower_round):
            torch.testing.assert_close(leader, follower, rtol=0, atol=0)


def test_tp1_keeps_acceptance_storage_and_skips_packet_allocation():
    group = SimpleNamespace(world_size=1, rank_in_group=0)
    tp = DSpineTpCoordinator(group)
    values = (torch.tensor([3], dtype=torch.int32), torch.tensor([2**40]), None)
    actual = tp.sync_accept(*values)
    assert all(left is right for left, right in zip(actual, values))
    assert not tp._accept_packets
    worker = _worker(group)
    result = (*values[:2], torch.tensor([7]), torch.tensor([[7]]), None)
    with patch.object(dflash.DFlashWorkerV2, "_accept_block", return_value=result):
        assert worker._accept_block(**_inputs(0)) is result
    assert not worker._dspine_tp._accept_packets
