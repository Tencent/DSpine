# Copyright 2026 Tencent
# Licensed under the Apache License, Version 2.0.

"""Real CPU/Gloo checks for original-vocabulary padding and selected-row exchange."""

import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang_dspine.sampling.tp_head import TPHead


class _Group:
    world_size = 2

    def all_gather(self, value, dim):
        result = [torch.empty_like(value) for _ in range(self.world_size)]
        dist.all_gather(result, value.contiguous())
        return torch.cat(result, dim=dim)

    def all_reduce(self, value):
        dist.all_reduce(value)
        return value


def _distributed_check(rank, rendezvous):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2
    )
    try:
        full = torch.arange(7 * 8).reshape(7, 8).to(torch.bfloat16)
        start, end = ((0, 4), (4, 7))[rank]
        shard = torch.full((4, 8), -900, dtype=torch.bfloat16)
        shard[: end - start] = full[start:end]
        head = TPHead(shard, start, end, 7, _Group())
        ids = torch.tensor([[0, 6, 4], [5, 1, 3]], dtype=torch.int64)
        torch.testing.assert_close(head.selected_rows(ids), full[ids], rtol=0, atol=0)
        table = (full[start:end].float()[:, :3] / 7).double()
        torch.testing.assert_close(
            head.gather_vocab_rows(table),
            (full.float()[:, :3] / 7).double(),
            rtol=0,
            atol=0,
        )
        values = torch.tensor([[9, 8, 8]], dtype=torch.bfloat16)
        local_ids = torch.tensor([[0, 1, 2] if rank == 0 else [4, 5, 6]])
        top_values, top_ids = head.global_topk(values, local_ids, 4)
        assert top_ids.tolist() == [[0, 4, 1, 2]], top_ids
        assert top_values.tolist() == [[9, 9, 8, 8]]
        rows = torch.ones((2, 8), dtype=torch.bfloat16)
        expected = (
            (full[ids].float() * rows[:, None].float()).sum(-1).to(torch.bfloat16)
        )
        torch.testing.assert_close(head.rescore(ids, rows), expected, rtol=0, atol=0)
        from sglang_dspine.kernels.dspine import serial_rs_table
        from sglang_dspine.sampling.factorized.cancellation import (
            Workspace as CancellationWorkspace,
        )
        from sglang_dspine.sampling.factorized.cancellation import (
            apply_direct_fallback,
        )
        from sglang_dspine.sampling.factorized.weights import build_cache

        torch.manual_seed(81)
        c, r, h, width, k = 6, 4, 8, 3, 3
        f = lambda *shape: torch.randn(*shape)
        code = f(7, c).bfloat16()
        pred, prev, bias, write, norm = f(r, c), f(r, c), f(r), f(h, r), f(h)
        emb, proj = f(7, 2).bfloat16(), f(r, 2)
        cache = build_cache(
            code, pred, prev, bias, write, norm, shard, emb, proj, tp_head=head
        )
        reference = build_cache(code, pred, prev, bias, write, norm, full, emb, proj)
        for name in ("G", "V", "TG", "Gram"):
            torch.testing.assert_close(
                cache[name], reference[name], rtol=1e-12, atol=1e-12
            )
        raw, query = f(width, h).bfloat16(), f(width, r).bfloat16()
        rms, anchor = torch.rand(width), torch.tensor([6])
        candidate_head = head.selected_rows(ids)
        common = (raw, rms, query, anchor, ids, code, pred, prev, bias, write, norm)
        baseline = serial_rs_table(
            *common, full, 1e-6, width, res_emb=emb, res_proj=proj
        )
        actual = serial_rs_table(
            *common,
            shard,
            1e-6,
            width,
            res_emb=emb,
            res_proj=proj,
            candidate_head=candidate_head,
        )
        torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
        x, a, message = f(2, h).double(), torch.rand(2).double(), f(2, k, r).double()
        safe = torch.zeros((2, k), dtype=torch.bool)
        direct, expected_direct = torch.empty(2, k, k), torch.empty(2, k, k)
        workspace = CancellationWorkspace.allocate(2, k, h, "cpu")
        apply_direct_fallback(
            x,
            a,
            message,
            ids,
            cache,
            safe,
            direct,
            workspace,
            1e-6,
            candidate_head=candidate_head,
        )
        apply_direct_fallback(
            x, a, message, ids, reference, safe, expected_direct, workspace, 1e-6
        )
        torch.testing.assert_close(direct, expected_direct, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


class TestDSpineTPHead(unittest.TestCase):
    def test_gloo_two_rank_vocabulary_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _distributed_check,
                args=(str(Path(directory) / "rendezvous"),),
                nprocs=2,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
