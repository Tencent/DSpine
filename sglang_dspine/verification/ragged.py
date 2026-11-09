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
        from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

        if self.full != 16:
            raise ValueError("Ragged verification requires draft block16")
        self.tokens = self.policy.batch8_tokens
        self.lens_cpu = [self.tokens // 8] * 8
        self.layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=self.lens_cpu,
            device=worker.device,
            grid=list(range(16, 129, 16)),
        )
        self.window_index = torch.tensor(
            [b * 16 + i for (b, n) in enumerate(self.lens_cpu) for i in range(n)],
            device=worker.device,
            dtype=torch.int64,
        )
        self.lens_tensor_cpu = torch.tensor(self.lens_cpu, dtype=torch.int64)
        with ragged_scope():
            old_init = self.runner.init_new_workspace
            try:
                self.ragged_backend = self.runner._get_attention_backend(
                    init_new_workspace=True
                )
            finally:
                self.runner.init_new_workspace = old_init
            # GB10: triton has no ragged verify graph; batches of 8 then verify full blocks
            self.ragged_supported = self.ragged_backend.supports_ragged_verify_graph
            self.ragged_graph = (
                None
                if not self.ragged_supported
                else DecodeCudaGraphRunner(
                    self.runner,
                    attn_backend=self.ragged_backend,
                    speculative_num_steps=1,
                    speculative_num_draft_tokens=16,
                )
                if self.states[16][1] is not None
                else None
            )
        self.executed_tokens = None
        self.ragged_round = False
        self.max_prefix = None
        logger.info(
            "DSPine ragged capture ready: lens=%s tokens=%d graph=%s",
            self.lens_cpu,
            self.tokens,
            self.ragged_graph is not None,
        )

    def begin(self, batch):
        super().begin(batch)
        self.executed_tokens = None
        self.ragged_round = (
            self.ragged_supported
            and self.in_round
            and batch.batch_size() == 8
            and self.tokens < 128
        )
        if self.ragged_round:
            self.active = self.full - 1

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
            self.ragged_graph,
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
            self.executed_tokens = rows
            long_prefix = (
                self.max_prefix >= 8192 if self.max_prefix is not None else None
            )
            key = (8, rows, long_prefix)
            self.executed_counts[key] += 1
            if self.executed_counts[key] == 1:
                logger.info(
                    "DSPine ragged executed: batch=8 draft=16 actual_rows=%d lens=%s cuda_graph=%s max_prefix=%s long_prefix=%s",
                    rows,
                    self.lens_cpu,
                    result.can_run_cuda_graph,
                    self.max_prefix,
                    long_prefix,
                )
            for name in ("next_token_logits", "hidden_states"):
                value = getattr(output, name)
                if value is not None:
                    if name == "next_token_logits":
                        from sglang_dspine.sampling.padding import logits_padding

                        padded = logits_padding(
                            self,
                            value,
                            (8 * 16, value.shape[-1]),
                        )
                    else:
                        padded = value.new_zeros((8 * 16, value.shape[-1]))
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
        if bs != 8 or self.executed_tokens != self.tokens:
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
            )
        top1 = next_token_logits.argmax(-1).view(bs, 16)
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
            T=16,
        )
        return (accept, commit, bonus, output, new_seq)
