"""DSpine serving on SGLang 0.5.22."""

import copy
import logging
import os
from contextlib import contextmanager

import torch
import triton
import triton.language as tl

from sglang_dspine.verification.caps import VerifyCaps

logger = logging.getLogger(__name__)
_depth = 0


def graph_tokens_for_budget(graph, budget, bs):
    """Use only an exact token tier with enough request slots; otherwise eager."""
    if graph is None:
        return None
    return next(
        (
            tier for tier in graph.capture_num_tokens
            if tier == budget and graph._ragged_capture_slots(tier) >= bs
        ),
        None,
    )


def ragged_active():
    return _depth > 0


@contextmanager
def ragged_scope():
    global _depth
    previous = os.environ.get("SGLANG_RAGGED_VERIFY_MODE")
    os.environ["SGLANG_RAGGED_VERIFY_MODE"] = "compact"
    _depth += 1
    try:
        yield
    finally:
        _depth -= 1
        if previous is None:
            os.environ.pop("SGLANG_RAGGED_VERIFY_MODE", None)
        else:
            os.environ["SGLANG_RAGGED_VERIFY_MODE"] = previous


@triton.jit
def _accept(
    Candidates,
    Top1,
    Lens,
    Accept,
    Commit,
    Bonus,
    Output,
    Prefix,
    NewSeq,
    T: tl.constexpr,
):
    b = tl.program_id(0)
    i = tl.arange(0, T)
    cap = tl.load(Lens + b)
    proposed = tl.load(Candidates + b * T + i + 1, i < T - 1, 0)
    predicted = tl.load(Top1 + b * T + i)
    matched = (i < cap - 1) & (proposed == predicted)
    accepted = tl.minimum(tl.min(tl.where(matched, T, i), 0), cap - 1)
    bonus = tl.load(Top1 + b * T + accepted)
    tl.store(Accept + b, accepted)
    tl.store(Commit + b, accepted + 1)
    tl.store(Bonus + b, bonus)
    tl.store(Output + b * T + i, tl.where(i == accepted, bonus, proposed))
    tl.store(NewSeq + b, tl.load(Prefix + b) + accepted + 1)


class RaggedVerifyCaps(VerifyCaps):
    def __init__(self, worker):
        super().__init__(worker)
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )
        if self.full != 16:
            raise ValueError("Ragged verification requires draft block16")
        self.tokens = self.batch_size = 0
        self.lens_cpu = self.lens_tensor_cpu = None
        self.ragged_backend = self.ragged_graph = None
        self.round_graph = None
        self.graph_tokens = None
        self.executed_tokens = None
        self.ragged_round = False
        self.compact_round = False
        self.compact_logits = None
        self.row_to_request = self.row_offsets = None
        self.max_prefix = None
        if not any(
            budget < bs * self.full
            for mode in ("greedy", "sampled")
            for bs, budget in self.policy_for_mode(mode).token_budgets.items()
        ):
            return
        with ragged_scope():
            old_init = self.runner.init_new_workspace
            try:
                self.ragged_backend = self.runner._get_attention_backend(
                    init_new_workspace=True
                )
            finally:
                self.runner.init_new_workspace = old_init
            if (
                not self.ragged_backend.supports_ragged_verify_graph
                or self.ragged_backend.needs_cpu_seq_lens
                or self.states[self.full][0].needs_cpu_seq_lens
            ):
                logger.info("DSPine SPS uses full-block verification on this backend")
                self.ragged_backend = None
                return
            self.ragged_graph = (
                DecodeCudaGraphRunner(
                    self.runner,
                    attn_backend=self.ragged_backend,
                    speculative_num_steps=1,
                    speculative_num_draft_tokens=self.full,
                )
                if self.states[self.full][1] is not None
                else None
            )
        logger.info("DSPine ragged capture ready: graph=%s", self.ragged_graph is not None)

    def begin(self, batch):
        super().begin(batch)
        self.executed_tokens = None
        self.batch_size = batch.batch_size() if self.in_round else 0
        self.tokens = self.policy.budget(self.batch_size, self.full)
        self.graph_tokens = graph_tokens_for_budget(
            self.ragged_graph, self.tokens, self.batch_size
        )
        self.round_graph = self.ragged_graph if self.graph_tokens is not None else None
        self.ragged_round = (
            self.in_round
            and self.ragged_backend is not None
            and self.tokens < self.batch_size * self.full
        )
        if self.ragged_round:
            self.active = self.full - 1
        from sglang_dspine.sampling.compact import can_keep_compact

        self.compact_round = bool(
            self.ragged_round and self.sampled_round
            and can_keep_compact(batch.sampling_info)
        )
        self.compact_logits = None
        self.row_to_request = self.row_offsets = None

    def finish(self):
        try:
            super().finish()
        finally:
            self.compact_round = False
            self.compact_logits = None
            self.row_to_request = self.row_offsets = None

    def prepare(self, original, verify_input, batch, target_worker, *args, **kwargs):
        if not self.in_round or not self.ragged_round:
            return super().prepare(
                original, verify_input, batch, target_worker, *args, **kwargs
            )
        if (
            verify_input.ragged_verify_layout is not None
            or verify_input.custom_mask is not None
        ):
            raise ValueError("Ragged DSPine requires an ordinary causal proposal")
        names = (
            "draft_token",
            "positions",
            "ragged_verify_layout",
            "kv_loc_cols",
            "live_seq_lens_cpu",
        )
        live = verify_input.live_seq_lens_cpu
        self.max_prefix = (
            int(live.max()) if live is not None and live.device.type == "cpu" else None
        )
        self.saved_input = (
            verify_input,
            {n: getattr(verify_input, n) for n in names},
            batch,
            batch.out_cache_loc,
        )
        verify_input.draft_token = verify_input.draft_token.index_select(
            0, self.window_index
        )
        verify_input.positions = verify_input.positions.index_select(
            0, self.window_index
        )
        verify_input.ragged_verify_layout = self.layout
        verify_input.kv_loc_cols = self.window_index
        batch.out_cache_loc = batch.out_cache_loc.index_select(0, self.window_index)
        (saved_cpu, saved_sum) = (batch.seq_lens_cpu, batch.seq_lens_sum)
        if getattr(self, "device_only_active", False):
            if self.ragged_backend.needs_cpu_seq_lens:
                raise RuntimeError("Cannot omit a required target CPU length mirror")
            batch.seq_lens_cpu = None
            batch.seq_lens_sum = None
            verify_input.live_seq_lens_cpu = None
        elif saved_cpu is not None:
            batch.seq_lens_cpu = (
                saved_cpu - self.full + self.lens_tensor_cpu.to(saved_cpu.dtype)
            )
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
        (self.runner.attn_backend, self.runner.decode_cuda_graph_runner) = (
            self.ragged_backend,
            self.round_graph,
        )
        try:
            with ragged_scope():
                return original(verify_input, batch, target_worker, *args, **kwargs)
        finally:
            (batch.seq_lens_cpu, batch.seq_lens_sum) = (saved_cpu, saved_sum)

    def target(self, original, target_worker, *args, **kwargs):
        if not self.in_round or not self.ragged_round or (not kwargs.get("is_verify")):
            return super().target(original, target_worker, *args, **kwargs)
        try:
            with ragged_scope():
                result = original(target_worker, *args, **kwargs)
            result = copy.copy(result)
            result.logits_output = copy.copy(result.logits_output)
            output = result.logits_output
            rows = output.next_token_logits.shape[0]
            if rows != self.tokens:
                raise RuntimeError(
                    f"Ineffective ragged verify: expected {self.tokens}, got {rows}"
                )
            self.executed_tokens = self.tokens
            long_prefix = (
                self.max_prefix >= 8192 if self.max_prefix is not None else None
            )
            key = (self.batch_size, rows, long_prefix)
            self.executed_counts[key] += 1
            if self.executed_counts[key] == 1:
                logger.info(
                    "DSPine ragged executed: batch=%d draft=%d actual_rows=%d lens=%s cuda_graph=%s max_prefix=%s long_prefix=%s",
                    self.batch_size,
                    self.full,
                    rows,
                    self.lens_cpu,
                    result.can_run_cuda_graph,
                    self.max_prefix,
                    long_prefix,
                )
            for name in ("next_token_logits", "hidden_states"):
                value = getattr(output, name)
                if value is not None:
                    if name == "next_token_logits" and self.compact_round:
                        self.compact_logits = value
                        self.row_offsets = self.layout.qo_indptr_device[:-1]
                        self.row_to_request = self.window_index.div(
                            self.full, rounding_mode="floor"
                        )
                        continue
                    if name == "next_token_logits":
                        from sglang_dspine.sampling.padding import logits_padding

                        padded = logits_padding(
                            self,
                            value,
                            (self.batch_size * self.full, value.shape[-1]),
                        )
                    else:
                        padded = value.new_zeros(
                            (self.batch_size * self.full, value.shape[-1])
                        )
                    padded.index_copy_(0, self.window_index, value)
                    setattr(output, name, padded)
            return result
        finally:
            self.restore_input()

    def accept(
        self,
        *,
        candidates,
        next_token_logits,
        sampling_info,
        draft_input,
        prefix_lens,
        bs,
    ):
        if not self.ragged_round:
            return super().accept(
                candidates=candidates,
                next_token_logits=next_token_logits,
                sampling_info=sampling_info,
                draft_input=draft_input,
                prefix_lens=prefix_lens,
                bs=bs,
            )
        if bs != self.batch_size or self.executed_tokens != self.tokens:
            raise RuntimeError(
                "Ragged acceptance requires confirmed packed verification"
            )
        if self.sampled_round:
            from sglang_dspine.sampling.rejection import accept_sampled_prefix

            return accept_sampled_prefix(
                self,
                candidates=candidates,
                next_token_logits=next_token_logits,
                sampling_info=sampling_info,
                prefix_lens=prefix_lens,
                bs=bs,
                verify_lens=self.layout.verify_lens,
                row_offsets=self.row_offsets if self.compact_round else None,
                row_to_request=self.row_to_request if self.compact_round else None,
            )
        top1 = next_token_logits.argmax(-1).view(bs, self.full)
        (accept, commit, bonus, output, new_seq) = (
            self.worker._next_accept_bonus_buffers(bs)
        )
        _accept[bs,](
            candidates,
            top1,
            self.layout.verify_lens,
            accept,
            commit,
            bonus,
            output,
            prefix_lens,
            new_seq,
            T=self.full,
        )
        return (accept, commit, bonus, output, new_seq)
