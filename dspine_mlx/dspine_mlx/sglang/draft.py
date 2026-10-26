"""Draft-side runtime for the DSpine MLX plugin.

Wraps the standalone ``dspine_mlx`` model/tail so the SGLang worker only deals
with per-request caches and sampling parameters.
"""

from __future__ import annotations

import os
import time
from contextlib import nullcontext
from dataclasses import dataclass

import mlx.core as mx

from ..bridge import build_serial_tables
from ..model import DSpineDraftModel, load_draft, quantize_draft
from ..propose import RefinementTail
from ..target import HeadAccess, _find_lm_head

from . import batched_tail
from .batch import DraftBatchContext
from .batched_attention import forward_block_batched
from .kernels import dispatch
from .kernels.config import batch_limits
from .sampling import RequestSampling


@dataclass
class DraftProposal:
    drafts: mx.array
    candidates: mx.array
    q_rows: mx.array | None


@dataclass
class BatchedProposal:
    """One decode round's proposals for every request in the batch."""

    drafts: mx.array          # [B, T-1] int32
    candidates: mx.array      # [B, T-1, K] int32
    q_rows: mx.array | None   # [B, T-1, K] fp32 for sampled requests
    greedy: mx.array          # [B] bool


class DraftRuntime:
    """Loads the DSpine draft and exposes per-request proposal helpers."""

    def __init__(self, draft, cfg, target_model, *, load_seconds: float = 0.0):
        self.draft: DSpineDraftModel = draft
        self.cfg = cfg
        self.block_size = int(cfg.block_size)
        self.trained_block_size = int(cfg.trained_block_size)
        self.mask_token_id = int(cfg.mask_token_id)
        self.head_topk = int(cfg.head_topk)
        self.vocab_size = int(cfg.vocab_size)
        self.load_seconds = float(load_seconds)

        self.draft.bind(target_model)
        self.head = HeadAccess(_find_lm_head(target_model))
        bridge = self.draft.selected_prefix_bridge
        t0 = time.perf_counter()
        self.tg, self.tsv = build_serial_tables(
            head_rows=self.head.rows_range,
            vocab_size=self.head.vocab_size,
            norm_w=self.draft.norm.weight,
            write_w=bridge.feature_write_weights[-1],
            pv_w=bridge.packed_weight[bridge.rank :],
            codebook=bridge.codebook,
        )
        self.tables_seconds = time.perf_counter() - t0
        self.tail = RefinementTail(self.draft, self.head, self.tg, self.tsv)

    @classmethod
    def load(cls, draft_path: str, target_model) -> "DraftRuntime":
        t0 = time.perf_counter()
        draft, cfg = load_draft(draft_path)
        bits = os.environ.get("DSPINE_MLX_DRAFT_BITS")
        if bits and not os.environ.get("DSPINE_MLX_NO_DRAFT_QUANT"):
            draft = quantize_draft(
                draft,
                bits=int(bits),
                group_size=int(os.environ.get("DSPINE_MLX_DRAFT_GROUP_SIZE", "64")),
                mode=os.environ.get("DSPINE_MLX_DRAFT_Q_MODE", "affine"),
            )
        return cls(
            draft,
            cfg,
            target_model,
            load_seconds=time.perf_counter() - t0,
        )

    def set_block_size(self, block_size: int) -> None:
        self.draft.set_block_size(block_size)
        self.block_size = int(block_size)

    def make_caches(self) -> list:
        return self.draft.make_cache()

    def append_context(self, hidden: mx.array, caches: list) -> None:
        """Append target hidden rows to the draft context K/V cache."""
        self.draft.append_context(hidden, caches)

    def propose(
        self,
        caches: list,
        anchor: int,
        block_size: int,
        sampling: RequestSampling,
        rng_key: mx.array,
    ) -> DraftProposal:
        """Draft one block for one request.

        ``caches`` must already contain the accepted context.  ``rng_key`` is
        consumed only for sampled requests.
        """
        T = int(block_size)
        rows, stash = self.run_backbone(caches, anchor, T)
        if sampling.greedy:
            drafts, cand = self.tail.greedy(rows, stash, int(anchor))
            return DraftProposal(drafts=drafts, candidates=cand, q_rows=None)

        uniforms = mx.random.uniform(key=rng_key, shape=(T - 1,))
        drafts, q_rows, cand = self.tail.sampled(
            rows,
            stash,
            int(anchor),
            inv_temp=sampling.inv_temp,
            top_k=sampling.chain_top_k,
            top_p=sampling.top_p,
            min_p=sampling.min_p,
            uniforms=uniforms,
        )
        return DraftProposal(drafts=drafts, candidates=cand, q_rows=q_rows)

    def run_backbone(self, caches: list, anchor: int, block_size: int):
        """One request's block-only draft forward; returns ``(rows, stash)``."""
        T = int(block_size)
        block = mx.array(
            [[int(anchor)] + [self.mask_token_id] * (T - 1)],
            dtype=mx.int32,
        )
        out, stash = self.draft(None, block, caches)
        return out[0, 1:], stash

    def propose_batch(
        self,
        caches: list[list],
        anchors: list[int],
        block_size: int,
        samplings: list[RequestSampling],
        rng_keys: list[mx.array],
        *,
        profiler=None,
    ) -> BatchedProposal:
        """Draft one block per request, batching the refinement tail.

        The backbone still runs per request (``B2`` batching is a separate,
        flag-gated path); the tail — head GEMM, candidate top-k, table and
        serial chain — runs once for the whole batch.
        """
        T = int(block_size)
        batch = len(anchors)
        if not (
            len(caches) == len(samplings) == len(rng_keys) == batch
        ):
            raise ValueError("propose_batch inputs are not aligned")
        phase = profiler.phase if profiler is not None else (lambda _n: nullcontext())

        with phase("draft.backbone"):
            if self._can_batch_backbone(caches, T):
                block_ids = mx.full(
                    (batch, T), self.mask_token_id, dtype=mx.int32
                )
                block_ids[:, 0] = mx.array(anchors, dtype=mx.int32)
                batch_ctx = DraftBatchContext.from_caches(caches, T)
                out, stash = forward_block_batched(
                    self.draft, block_ids, batch_ctx
                )
                rows = out[:, 1:, :].reshape(batch * (T - 1), -1)
            else:
                rows_list = []
                parts = {"raw": [], "rms": [], "packed": []}
                for i in range(batch):
                    rows_i, stash_i = self.run_backbone(caches[i], anchors[i], T)
                    rows_list.append(rows_i)
                    for key in parts:
                        parts[key].append(stash_i[key])
                rows = mx.concatenate(rows_list, axis=0)
                stash = {
                    key: mx.concatenate(vals, axis=0)
                    for key, vals in parts.items()
                }

        anchors_arr = mx.array([int(a) for a in anchors], dtype=mx.int32)
        greedy = [bool(s.greedy) for s in samplings]
        greedy_mask = mx.array(greedy)
        inv = mx.array([float(s.inv_temp) for s in samplings], dtype=mx.float32)
        topk = mx.array([int(s.chain_top_k) for s in samplings], dtype=mx.int32)
        topp = mx.array([float(s.top_p) for s in samplings], dtype=mx.float32)
        minp = mx.array([float(s.min_p) for s in samplings], dtype=mx.float32)
        uniforms = mx.concatenate(
            [mx.random.uniform(key=k, shape=(T - 1,)) for k in rng_keys]
        )

        with phase("tail.batch"):
            if all(greedy):
                out = batched_tail.greedy_batched(
                    self.tail, rows, stash, anchors_arr, batch=batch, block=T
                )
            elif not any(greedy):
                out = batched_tail.sampled_batched(
                    self.tail,
                    rows,
                    stash,
                    anchors_arr,
                    inv_temp=inv,
                    top_k=topk,
                    top_p=topp,
                    min_p=minp,
                    uniforms=uniforms,
                    batch=batch,
                    block=T,
                )
            else:
                out = batched_tail.forward_batched(
                    self.tail,
                    rows,
                    stash,
                    anchors_arr,
                    greedy_mask=greedy_mask,
                    inv_temp=inv,
                    top_k=topk,
                    top_p=topp,
                    min_p=minp,
                    uniforms=uniforms,
                    batch=batch,
                    block=T,
                )
        return BatchedProposal(
            drafts=out.tokens,
            candidates=out.candidates,
            q_rows=out.q_rows,
            greedy=greedy_mask,
        )

    def _can_batch_backbone(self, caches: list[list], block_size: int) -> bool:
        """B2 is skipped for single requests and very long/ragged contexts."""
        if not dispatch.draft_batch_enabled() or len(caches) < 2:
            return False
        max_prefix, max_spread = batch_limits()
        lens = [int(c[0].offset) for c in caches]
        return max(lens) <= max_prefix and (max(lens) - min(lens)) <= max_spread
