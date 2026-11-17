# Copyright 2026 Tencent
# Licensed under the Apache License, Version 2.0.

"""Distributed candidate retrieval and unchanged refinement on selected head rows."""

import tempfile
import traceback
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


class _Group:
    @property
    def world_size(self):
        return dist.get_world_size()

    def all_gather(self, value, dim):
        result = [torch.empty_like(value) for _ in range(self.world_size)]
        dist.all_gather(result, value.contiguous())
        return torch.cat(result, dim=dim)

    def all_reduce(self, value):
        dist.all_reduce(value)
        return value


def _cpu_topk_ids(scores, k):
    return torch.tensor(
        [
            sorted(range(len(row)), key=lambda token: (-row[token], token))[:k]
            for row in scores.cpu().float().tolist()
        ],
        device=scores.device,
    )


def _forbidden_flashinfer(*args, **kwargs):
    raise AssertionError("FlashInfer must not run on the portable path")


def _cuda_worker(rank, rendezvous, world_size, portable):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size
    )
    try:
        from flashinfer.topk import top_k

        from sglang_dspine.kernels import dspine as kernels
        from sglang_dspine.layers.fp8 import coarse as coarse_ops
        from sglang_dspine.layers.fp8 import gather as gather_ops
        from sglang_dspine.layers.fp8.coarse import FullVFP8Coarse, stable_local_topk
        from sglang_dspine.sampling.factorized.cancellation import (
            Workspace as CancellationWorkspace,
        )
        from sglang_dspine.sampling.factorized.cancellation import (
            apply_direct_fallback,
        )
        from sglang_dspine.sampling.factorized.kernels import Workspace, factor_table
        from sglang_dspine.sampling.factorized.weights import build_cache
        from sglang_dspine.sampling.tp_head import TPHead, merge_topk

        if portable:
            coarse_ops.supports_filtered_topk = lambda device: False
            top_k = _forbidden_flashinfer

        torch.manual_seed(884)
        torch.backends.cuda.matmul.allow_tf32 = False
        # Original-vocabulary rows exclude the final shard's physical padding.
        v, h, c, r, bs, width, k = 510, 512, 16, 128, 2, 4, 16
        physical_rows = 512 // world_size
        start, end = rank * physical_rows, min(v, (rank + 1) * physical_rows)
        full = torch.randn(v, h, device="cuda", dtype=torch.bfloat16) / 8
        shard = torch.zeros(physical_rows, h, device="cuda", dtype=torch.bfloat16)
        shard[: end - start].copy_(full[start:end])
        head = TPHead(shard, start, end, v, _Group())
        n = bs * (width - 1)
        hidden = torch.randn(n, h, device="cuda", dtype=torch.bfloat16)
        selected_ids = torch.tensor([[0, 256, 509]], device="cuda")
        selected = head.selected_rows(selected_ids)
        torch.testing.assert_close(selected, full[selected_ids], rtol=0, atol=0)
        # Same coarse matmul per physical shard -> independent full-vector reference.
        local = FullVFP8Coarse(torch, shard, head)
        result, coarse_ids, _ = local.retrieve_and_rescore(
            hidden, fi_top_k=top_k, gather_ops=gather_ops
        )
        def coarse_reference():
            scale = hidden.float().abs().amax(-1, keepdim=True) / 448
            q = (hidden.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            scores = torch._scaled_mm(
                q,
                local.weight8_t,
                scale,
                local.weight_scale,
                out_dtype=torch.bfloat16,
                use_fast_accum=False,
            )
            scores[:, end - start :] = -torch.inf
            all_scores = head.gather_vocab_rows(scores[:, : end - start].T).T.contiguous()
            if portable:
                return _cpu_topk_ids(all_scores, 128)
            global_ids = torch.arange(v, device="cuda").expand(n, v)
            return merge_topk(all_scores, global_ids, 128)[1]

        expected_coarse = coarse_reference()
        torch.testing.assert_close(coarse_ids, expected_coarse, rtol=0, atol=0)
        expected = gather_ops.select16(
            torch,
            gather_ops.rescore(torch, full, expected_coarse, hidden),
            expected_coarse,
        )
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        if portable:
            for _ in range(3):
                local.retrieve_and_rescore(
                    hidden, fi_top_k=top_k, gather_ops=gather_ops
                )
            torch.cuda.synchronize()
            coarse_graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(coarse_graph):
                    replay_result, replay_coarse, _ = local.retrieve_and_rescore(
                        hidden, fi_top_k=top_k, gather_ops=gather_ops
                    )
                for _ in range(2):
                    hidden.neg_()
                    expected_coarse = coarse_reference()
                    expected = gather_ops.select16(
                        torch,
                        gather_ops.rescore(torch, full, expected_coarse, hidden),
                        expected_coarse,
                    )
                    coarse_graph.replay()
                    torch.testing.assert_close(
                        replay_coarse, expected_coarse, rtol=0, atol=0
                    )
                    torch.testing.assert_close(replay_result, expected, rtol=0, atol=0)
                torch.cuda.synchronize()
            finally:
                coarse_graph.reset()
        for tied in (
            torch.zeros(n, 256, device="cuda", dtype=torch.bfloat16),
            torch.arange(256, device="cuda")
            .remainder(3)
            .bfloat16()
            .expand(n, 256)
            .contiguous(),
        ):
            for local_k in (16, 128):
                values, ids = stable_local_topk(tied, local_k, top_k)
                expect_ids = _cpu_topk_ids(tied, local_k)
                torch.testing.assert_close(
                    ids.sort(-1).values, expect_ids.sort(-1).values, rtol=0, atol=0
                )
                torch.testing.assert_close(values, tied.gather(-1, ids), rtol=0, atol=0)
        f = lambda *shape: torch.randn(*shape, device="cuda")
        code, norm, write = f(v, c).bfloat16(), f(h).bfloat16(), f(h, r).bfloat16()
        pred, prev, bias = f(r, c), f(r, c), f(r)
        packed_w = torch.cat((pred, prev)).bfloat16()
        tg_local, tsv_local = kernels.build_serial_tables(
            shard[: end - start], norm, write, packed_w, code[start:end]
        )
        tg, tsv = head.gather_vocab_rows(tg_local), head.gather_vocab_rows(tsv_local)
        ref_tg, ref_tsv = kernels.build_serial_tables(full, norm, write, packed_w, code)
        torch.testing.assert_close(tg, ref_tg, rtol=0, atol=0)
        torch.testing.assert_close(tsv, ref_tsv, rtol=0, atol=0)
        raw, query = f(bs * width, h).bfloat16(), f(bs * width, r).bfloat16()
        rms = torch.rand(bs * width, device="cuda")
        anchors = torch.tensor([1, 509], device="cuda", dtype=torch.int64)
        inputs = torch.zeros(bs, width, device="cuda", dtype=torch.int64)
        inputs[:, 0] = anchors
        candidate_head = head.selected_rows(result)
        args = (
            raw,
            norm,
            result,
            inputs.flatten(),
            full,
            tg,
            tsv,
            query,
            rms,
            bias,
            width,
        )
        confidence, ref_confidence = (
            torch.empty(n, k, device="cuda"),
            torch.empty(n, k, device="cuda"),
        )
        baseline = kernels.serial_table(*args, confidence=ref_confidence)
        actual = kernels.serial_table(
            *args[:4],
            shard,
            *args[5:],
            confidence=confidence,
            candidate_head=candidate_head,
        )
        torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
        torch.testing.assert_close(confidence, ref_confidence, rtol=0, atol=0)
        for residual_rank in (0, 2):
            emb, proj = (f(v, 2).bfloat16(), f(r, 2)) if residual_rank else (None, None)
            cache = build_cache(
                code, pred, prev, bias, write, norm, shard, emb, proj, tp_head=head
            )
            ref = build_cache(code, pred, prev, bias, write, norm, full, emb, proj)
            for name in ("G", "V", "TG", "Gram"):
                torch.testing.assert_close(
                    cache[name], ref[name], rtol=1e-12, atol=1e-12
                )
            out, ref_out = (
                torch.empty(n, k, k, device="cuda"),
                torch.empty(n, k, k, device="cuda"),
            )
            base = torch.empty(n, k, device="cuda", dtype=torch.float64)
            work = Workspace.allocate(n, k, h, r, "cuda")
            factor_table(
                raw, rms, query, anchors, result, ref, 1e-6, width, ref_out, base, work
            )
            factor_table(
                raw,
                rms,
                query,
                anchors,
                result,
                cache,
                1e-6,
                width,
                out,
                base,
                work,
                candidate_head=candidate_head,
            )
            torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
            # Exercise direct64 fallback on every row, independently of safety heuristics.
            x, a, message = (
                f(n, h).double(),
                torch.rand(n, device="cuda").double(),
                f(n, k, r).double(),
            )
            safe = torch.zeros(n, k, device="cuda", dtype=torch.bool)
            fallback = CancellationWorkspace.allocate(n, k, h, "cuda")
            apply_direct_fallback(
                x, a, message, result, ref, safe, ref_out, fallback, 1e-6
            )
            apply_direct_fallback(
                x,
                a,
                message,
                result,
                cache,
                safe,
                out,
                fallback,
                1e-6,
                candidate_head=candidate_head,
            )
            torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
        # Collective replay must refresh selected rows, not retain the first captured IDs.
        ids = result.clone()
        for _ in range(3):
            head.selected_rows(ids)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                replayed = head.selected_rows(ids)
            ids.copy_((ids + 257) % v)
            graph.replay()
            torch.testing.assert_close(replayed, full[ids], rtol=0, atol=0)
            torch.cuda.synchronize()
        finally:
            # NCCL cannot shut down while a captured graph owns its communicator.
            graph.reset()
        del graph, replayed
    except BaseException:
        # Emit the numerical failure before a peer's pending collective can delay teardown.
        traceback.print_exc()
        raise
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("world_size", "portable"),
    [(2, False), (4, False), (2, True)],
    ids=["tp2-native", "tp4-native", "tp2-portable"],
)
def test_sharded_head_and_refinement(world_size, portable):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"{world_size} CUDA devices required")
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(
            _cuda_worker,
            args=(str(Path(directory) / "rendezvous"), world_size, portable),
            nprocs=world_size,
            join=True,
        )
