"""The DSPINE speculative algorithm as SGLang's MLX backend sees it.

SGLang 0.5.18's MLX backend has no speculative-decoding worker, but its generic
scheduler already understands spec-v2 results.  This ``CustomSpecAlgo`` claims
``is_dflash_family()`` so ``ScheduleBatch.prepare_for_decode`` reserves verify
slots through ``DFlashDraftInputV2``; the actual draft/verify work is done by
:class:`dspine_mlx.sglang.worker.DSpineMlxWorker`.

v1 deliberately runs the synchronous (non-overlap) scheduler path: the MLX
overlap loop bypasses ``run_batch`` and cannot drive a spec worker.  Continuous
batching across requests still comes from the scheduler.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Type

import torch
from sglang.srt.arg_groups.overrides import resolved_view
from sglang.srt.mem_cache.allocation import alloc_token_slots
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.spec_registry import CustomSpecAlgo

ALGORITHM = "DSPINE"
logger = logging.getLogger(__name__)


@dataclass
class DSpineMlxDraftInput(DFlashDraftInputV2):
    """DFlash draft state with a CPU-only ``prepare_for_decode``.

    Upstream ``DFlashDraftInputV2.prepare_for_decode`` assigns slots through a
    Triton kernel (``assign_req_to_token_pool_func``), which does not exist on
    the MLX/macOS path.  The MLX stub pools are plain CPU torch tensors, so a
    direct slice assignment is equivalent and portable.
    """

    block_size: int | None = None

    def prepare_for_decode(self, batch) -> None:
        bs = batch.batch_size()
        if bs == 0:
            return
        block_size = int(self.block_size or 0)
        if block_size <= 0:
            raise ValueError("DSpineMlxDraftInput.block_size was not set")
        page_size = int(batch.token_to_kv_pool_allocator.page_size)
        if page_size != 1:
            raise ValueError(
                f"DSPINE MLX requires page_size == 1, got {page_size}."
            )

        committed: list[int] = []
        cur: list[int] = []
        reserved: list[int] = []
        num_needed = 0
        for req in batch.reqs:
            c = int(req.kv_committed_len)
            a = int(req.kv.kv_allocated_len)
            r = max(a, c + 2 * block_size)
            committed.append(c)
            cur.append(a)
            reserved.append(r)
            num_needed += r - a

        if num_needed > 0:
            out_cache_loc = alloc_token_slots(batch.tree_cache, num_needed)
            req_to_token = batch.req_to_token_pool.req_to_token
            offset = 0
            for i, req in enumerate(batch.reqs):
                n = reserved[i] - cur[i]
                if n > 0:
                    pool_idx = int(batch.req_pool_indices[i])
                    req_to_token[pool_idx, cur[i] : reserved[i]] = out_cache_loc[
                        offset : offset + n
                    ]
                offset += n
            for i, req in enumerate(batch.reqs):
                req.kv.kv_allocated_len = max(
                    int(req.kv.kv_allocated_len), reserved[i]
                )

        for req in batch.reqs:
            req.decode_batch_idx += 1
        batch.seq_lens_cpu = torch.tensor(committed, dtype=torch.int64)
        batch.seq_lens_sum = int(sum(committed))
        self.reserved_seq_lens_cpu = torch.tensor(reserved, dtype=torch.int32)
        self.reserved_seq_lens_sum = int(sum(reserved))


def make_draft_input(bonus_tokens, new_seq_lens, block_size: int):
    bs = int(new_seq_lens.numel())
    return DSpineMlxDraftInput(
        topk_p=torch.empty((bs, 0), dtype=torch.float32),
        topk_index=torch.empty((bs, 0), dtype=torch.int64),
        bonus_tokens=bonus_tokens.to(dtype=torch.int64),
        new_seq_lens=new_seq_lens.to(dtype=torch.int64),
        hidden_states=torch.empty((bs, 0), dtype=torch.float16),
        block_size=int(block_size),
    )


class DSpineMlxSpecAlgo(CustomSpecAlgo):
    def is_dflash_family(self) -> bool:
        return True

    def supports_target_verify_for_draft(self) -> bool:
        return True

    def supports_grammar_overlap(self) -> bool:
        # v1 verify does not advance the grammar FSM inside the forward.
        return False

    def need_topk(self) -> bool:
        return False

    def carries_draft_hidden_states(self) -> bool:
        return False

    def create_future_map(
        self,
        device,
        req_to_token_pool,
        needs_cpu_seq_lens: bool = True,
        needs_confidence_relay: bool = False,
    ):
        from sglang.srt.managers.overlap_utils import FutureMap

        return FutureMap(
            device, self, req_to_token_pool, needs_cpu_seq_lens, needs_confidence_relay
        )

    def create_worker(self, server_args) -> Type:
        if not bool(getattr(server_args, "disable_overlap_schedule", False)):
            raise ValueError(
                "DSPINE on the MLX backend requires --disable-overlap-schedule "
                "(the MLX overlap loop cannot drive a speculative worker)."
            )
        return self.factory(server_args)

    def handle_server_args(self, server_args) -> None:
        from sglang.srt.utils.tensor_bridge import use_mlx

        if not use_mlx():
            raise ValueError(
                "DSPINE was registered by the MLX plugin, but SGLANG_USE_MLX=1 "
                "is not set. For the CUDA backend install sglang_dspine instead."
            )
        cfg = resolved_view(server_args)
        device = str(getattr(cfg, "device", server_args.device))
        if device != "mps":
            raise ValueError(
                f"DSPINE MLX requires the MPS/MLX device, got device={device!r}."
            )
        _require(cfg, "tp_size", 1, "DSPINE MLX supports tp_size == 1 only.")
        _require(cfg, "pp_size", 1, "DSPINE MLX supports pp_size == 1 only.")
        _require(cfg, "dp_size", 1, "DSPINE MLX supports dp_size == 1 only.")
        if getattr(cfg, "enable_dp_attention", False):
            raise ValueError("DSPINE MLX does not support DP attention.")
        if getattr(cfg, "disaggregation_mode", "null") != "null":
            raise ValueError("DSPINE MLX does not support prefill/decode disaggregation.")
        if getattr(cfg, "speculative_draft_window_size", None) is not None:
            raise ValueError("DSPINE MLX does not support --speculative-draft-window-size.")
        if getattr(cfg, "enable_hierarchical_cache", False):
            raise ValueError("DSPINE MLX does not support hierarchical cache.")
        if getattr(cfg, "enable_two_batch_overlap", False):
            raise ValueError("DSPINE MLX does not support two-batch overlap.")
        if getattr(cfg, "speculative_adaptive", False):
            raise ValueError("DSPINE MLX does not support adaptive speculative decoding.")
        if getattr(cfg, "enable_linear_replayssm_spec", False):
            raise ValueError("DSPINE MLX does not support --enable-linear-replayssm-spec.")
        if getattr(cfg, "kv_cache_dtype", None) in ("nvfp4", "fp4_mx_block16"):
            raise ValueError("DSPINE MLX does not support FP4 KV cache.")
        if getattr(cfg, "lora_paths", None):
            raise ValueError("DSPINE MLX does not support LoRA.")

        draft_path = getattr(cfg, "speculative_draft_model_path", None)
        if not draft_path:
            raise ValueError(
                "DSPINE MLX requires --speculative-draft-model-path (a local "
                "directory or Hugging Face repo id such as "
                "tencent/Qwen3-8B-DSpine-b16)."
            )
        from ..config import load_draft_config, resolve_draft_path

        # Download a repo id once here; scheduler processes get the local dir.
        draft_path = str(
            resolve_draft_path(
                draft_path, getattr(cfg, "speculative_draft_model_revision", None)
            )
        )
        server_args.speculative_draft_model_path = draft_path
        draft_cfg = load_draft_config(draft_path)
        served = getattr(cfg, "speculative_num_draft_tokens", None)
        if served is None:
            served = int(draft_cfg.block_size)
        else:
            served = int(served)
        draft_cfg.validate_served_block_size(served)

        # v1 envelope: synchronous scheduler, no prefix cache, MLX sampling.
        server_args.disable_overlap_schedule = True
        server_args.enable_mixed_chunk = False
        server_args.disable_cuda_graph = True
        server_args.disable_radix_cache = True
        server_args.mlx_enable_sampling = True
        server_args.speculative_num_steps = 1
        server_args.speculative_eagle_topk = 1
        server_args.speculative_num_draft_tokens = served
        if getattr(server_args, "max_running_requests", None) is None:
            server_args.max_running_requests = 8

        logger.info(
            "DSPINE MLX: block_size=%d (trained %d), synchronous scheduler, "
            "radix cache disabled, MLX sampling enabled",
            served,
            int(draft_cfg.trained_block_size),
        )


def _require(cfg, name: str, expected, message: str) -> None:
    value = getattr(cfg, name, expected)
    if value != expected:
        raise ValueError(f"{message} (got {name}={value!r})")


def worker_factory(server_args) -> Type:
    from .worker import DSpineMlxWorker

    return DSpineMlxWorker
