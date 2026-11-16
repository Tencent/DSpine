# Copyright 2026 Tencent
# Licensed under the Apache License, Version 2.0.

"""Rank agreement of real DSpine RNG, proposals and packed layouts over Gloo."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang_dspine.runtime.tp_sync import DSpineTpCoordinator
from sglang_dspine.sampling.random import (
    ACCEPT,
    DRAFT,
    PREFILL,
    RequestRNG,
    seeded_batch,
    seeded_uniforms,
    sync_prefill_tokens,
    verification_uniforms,
)
from sglang_dspine.verification.sps import device_plan


class _GlooBatchGroup:
    """Real collectives, with local group leader deliberately global rank one."""

    world_size = 2
    ranks = (1, 0)

    def __init__(self, rank):
        self.rank_in_group = self.ranks.index(rank)

    def broadcast(self, value, src):
        dist.broadcast(value, src=self.ranks[src])

    def broadcast_object(self, value, src):
        values = [value]
        dist.broadcast_object_list(values, src=self.ranks[src])
        return values[0]


def _req(seed=None):
    return SimpleNamespace(sampling_params=SimpleNamespace(sampling_seed=seed))


def _batch(reqs):
    return SimpleNamespace(
        reqs=reqs,
        sampling_info=SimpleNamespace(
            sampling_seed=None, temperatures=torch.ones(len(reqs), 1)
        ),
    )


def _equal_on_ranks(value):
    outputs = [torch.empty_like(value) for _ in range(2)]
    dist.all_gather(outputs, value)
    torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)


def _distributed_worker(rank, rendezvous, receipt_dir):
    os.environ["SGLANG_SPEC_TP_SYNC"] = "off"
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2
    )
    try:
        torch.manual_seed(100 + rank)
        tp = DSpineTpCoordinator(_GlooBatchGroup(rank))
        seed_values = iter((101, 202, 303))

        def make_seed(_bits):
            assert tp.is_leader, "Only the batch TP group's leader creates seeds"
            return next(seed_values)

        rng = RequestRNG(tp)
        reqs = [_req(), _req()]
        batch = _batch(reqs)
        with patch(
            "sglang_dspine.runtime.tp_sync.secrets.randbits",
            side_effect=make_seed,
        ):
            with seeded_batch(batch, tp):
                assert batch.sampling_info._dspine_tp is tp
                assert batch.sampling_info.sampling_seed.tolist() == [101, 202]
                rng.stage(batch.sampling_info.sampling_seed, torch.tensor([8, 31]))
                _equal_on_ranks(rng.draw(3, DRAFT))
                _equal_on_ranks(rng.draw(3, ACCEPT))
                _equal_on_ranks(rng.draw(1, PREFILL))
            assert batch.sampling_info.sampling_seed is None
            assert not hasattr(batch.sampling_info, "_dspine_tp")
            mixed = _batch([_req(99), _req(), reqs[0]])
            with seeded_batch(mixed, tp):
                assert mixed.sampling_info.sampling_seed.tolist() == [99, 303, 101]
                expected = seeded_uniforms(
                    torch.tensor([99, 303, 101]), torch.tensor([3, 5, 8]), 3, DRAFT
                )
                rng.stage(mixed.sampling_info.sampling_seed, torch.tensor([3, 5, 8]))
                torch.testing.assert_close(rng.draw(3, DRAFT), expected, atol=0, rtol=0)
                # Identical seed, but P crosses that coin by one small rounding
                # margin. The actual prefill helper must still agree on anchor.
                coin = seeded_uniforms(
                    mixed.sampling_info.sampling_seed,
                    torch.tensor([3, 5, 8]),
                    1,
                    PREFILL,
                ).double()
                p0 = coin + (2.0**-20 if rank == 0 else -(2.0**-20))
                cdf = torch.cat((p0, torch.ones_like(p0)), dim=1)
                local_tokens = (cdf > coin).int().argmax(-1).int()
                assert local_tokens.tolist() == [rank] * 3
                anchors = sync_prefill_tokens(local_tokens, mixed.sampling_info)
                assert anchors.tolist() == [1] * 3
            assert mixed.sampling_info.sampling_seed is None
            with (
                patch(
                    "sglang_dspine.runtime.tp_sync.secrets.randbits",
                    side_effect=AssertionError("Existing request seed was regenerated"),
                ),
                seeded_batch(_batch([reqs[1], reqs[0]]), tp) as _unused,
            ):
                pass

        rng.stage(None, torch.tensor([8, 31]))
        before = torch.get_rng_state()
        coins, bonus_coins = verification_uniforms(2, 4, "cpu", rng)
        _equal_on_ranks(coins)
        _equal_on_ranks(bonus_coins)
        if not tp.is_leader:
            assert torch.equal(before, torch.get_rng_state())

        ids = torch.arange(24).reshape(2, 3, 4) + rank * 100
        probabilities = torch.tensor(
            [0.4, 0.3, 0.2, 0.1] if rank == 1 else [0.1, 0.2, 0.3, 0.4]
        )
        sampler = SimpleNamespace(
            block_size=4,
            out=ids[..., 0].reshape(-1).clone(),
            candidate_out=ids.clone(),
            q_out=probabilities.expand(2, 3, 4).clone(),
            confidence_out=torch.full((2, 3), 0.1 + rank * 0.4),
            round_sampled=True,
        )
        tp.sync_proposal(sampler, 2)
        for value in (
            sampler.out,
            sampler.candidate_out,
            sampler.q_out,
            sampler.confidence_out,
        ):
            _equal_on_ranks(value)
        assert sampler.out.tolist() == [100, 104, 108, 112, 116, 120]
        torch.testing.assert_close(
            sampler.q_out[0, 0], torch.tensor([0.4, 0.3, 0.2, 0.1])
        )

        raw = torch.tensor([[0.99] * 3, [0.5] * 3, [0.01] * 3])
        if rank:
            raw = raw.flip(0)
        table = torch.linspace(0.01, 0.99, 32).expand(3, 32)
        cfg = SimpleNamespace(min_verify_len=1, survival_eps=0.0)
        layout, index = device_plan(raw, table, torch.arange(3), 7, cfg, tp=tp)
        for value in (layout.verify_lens, layout.qo_indptr_device, index):
            _equal_on_ranks(value)
        assert index.tolist() == [
            row * 4 + col
            for row, length in enumerate(layout.verify_lens.tolist())
            for col in range(length)
        ]
        # A planner graph writes these same buffers; sync must retain pointers.
        local_layout, local_index = device_plan(raw, table, torch.arange(3), 7, cfg)
        values = (
            local_layout.verify_lens,
            local_layout.extend_start_loc,
            local_layout.qo_indptr_device,
            local_index,
        )
        pointers = [value.data_ptr() for value in values]
        tp.sync_layout(local_layout, local_index)
        assert pointers == [value.data_ptr() for value in values]
        for value in values:
            _equal_on_ranks(value)
        assert len(tp._layout_packets) == 1
        tp.sync_layout(local_layout, local_index)
        assert len(tp._layout_packets) == 1

        accepted, bonus = tp.sync_accept(
            torch.tensor([rank, rank + 1], dtype=torch.int32),
            torch.tensor([7, 9], dtype=torch.int64) + rank,
        )
        assert accepted.tolist() == [1, 2] and bonus.tolist() == [8, 10]
        _equal_on_ranks(torch.tensor([21, 32]) + accepted + 1)
        Path(receipt_dir, f"rank{rank}.json").write_text(
            json.dumps({"status": "PASS", "group_rank": tp.group.rank_in_group})
        )
    finally:
        dist.destroy_process_group()


class TestDSpineTp(unittest.TestCase):
    def test_real_two_process_rank_agreement(self):
        with tempfile.TemporaryDirectory(prefix="dspine-gloo-") as folder:
            mp.spawn(
                _distributed_worker,
                args=(str(Path(folder, "rendezvous")), folder),
                nprocs=2,
                join=True,
            )
            for rank in range(2):
                self.assertEqual(
                    json.loads(Path(folder, f"rank{rank}.json").read_text())["status"],
                    "PASS",
                )

    def test_tp1_unseeded_requests_keep_native_rng_path(self):
        group = SimpleNamespace(world_size=1, rank_in_group=0)
        tp = DSpineTpCoordinator(group)
        batch = _batch([_req(), _req()])
        with (
            patch(
                "sglang_dspine.sampling.random.secrets.randbits",
                side_effect=AssertionError(
                    "TP1 should not synthesize an unseeded request seed"
                ),
            ),
            seeded_batch(batch, tp),
        ):
            self.assertIsNone(batch.sampling_info.sampling_seed)
        self.assertFalse(any(hasattr(req, "_dspine_random_seed") for req in batch.reqs))

    def test_existing_explicit_seed_tensor_is_preserved(self):
        batch = _batch([_req(5)])
        original = torch.tensor([5])
        batch.sampling_info.sampling_seed = original
        previous_tp = object()
        batch.sampling_info._dspine_tp = previous_tp
        with seeded_batch(batch, SimpleNamespace(enabled=True)):
            self.assertIs(batch.sampling_info.sampling_seed, original)
        self.assertIs(batch.sampling_info.sampling_seed, original)
        self.assertIs(batch.sampling_info._dspine_tp, previous_tp)
