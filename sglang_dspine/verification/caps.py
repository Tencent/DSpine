"""DSpine serving on SGLang 0.5.22."""

import copy
import logging
from collections import Counter
from contextlib import contextmanager

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _accept_capped(
    Candidates,
    Top1,
    Accept,
    Commit,
    Bonus,
    Output,
    Prefix,
    NewSeq,
    T: tl.constexpr,
    CAP: tl.constexpr,
    WIDTH: tl.constexpr,
):
    b = tl.program_id(0)
    i = tl.arange(0, WIDTH)
    proposed = tl.load(Candidates + b * T + i + 1, i < T - 1, 0)
    predicted = tl.load(Top1 + b * T + i, i < T, 0)
    matched = (i < CAP - 1) & (proposed == predicted)
    accepted = tl.minimum(tl.min(tl.where(matched, T, i), 0), CAP - 1)
    bonus = tl.load(Top1 + b * T + accepted)
    tl.store(Accept + b, accepted)
    tl.store(Commit + b, accepted + 1)
    tl.store(Bonus + b, bonus)
    emitted = tl.where(i == accepted, bonus, proposed)
    tl.store(Output + b * T + i, emitted, i < T)
    prefix = tl.load(Prefix + b)
    tl.store(NewSeq + b, prefix + accepted + 1)


@contextmanager
def spec_width(width):
    from sglang.srt.runtime_context import get_context, get_spec

    old = (get_spec().speculative_num_steps, get_spec().speculative_num_draft_tokens)
    get_context().override(
        "dspine.verify_cap.enter",
        speculative_num_steps=1,
        speculative_num_draft_tokens=int(width),
    )
    try:
        yield
    finally:
        get_context().override(
            "dspine.verify_cap.exit",
            speculative_num_steps=old[0],
            speculative_num_draft_tokens=old[1],
        )


class VerifyCaps:
    def __init__(self, worker):
        from sglang.srt.model_executor.cuda_graph_config import Backend
        from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
            DecodeCudaGraphRunner,
        )
        from sglang.srt.runtime_context import get_exec, get_parallel, get_spec

        from sglang_dspine.verification.policy import SpsPolicy, hardware_identity

        if get_parallel().attn_dp_enabled or int(get_parallel().pp_size) != 1:
            raise ValueError(
                "DSPine prefix verification currently supports TP1/DP1/PP1"
            )
        self.worker = worker
        self.full = int(worker.block_size)
        self.policy = SpsPolicy.load(
            get_spec().speculative_dspark_sps_table_path,
            hardware_identity(worker.device),
        )
        self.widths = sorted({self.full, *self.policy.uniform_widths})
        logger.info("DSPine SPS policy: %s", self.policy.to_dict())
        self.runner = worker.target_worker.model_runner
        self.states = {
            self.full: (self.runner.attn_backend, self.runner.decode_cuda_graph_runner)
        }
        self.active = self.full
        self.in_round = False
        self.saved_input = None
        self.counts = Counter()
        self.executed_counts = Counter()
        self.executed_width = None
        self.sampled_executions = 0
        self.sampled_round = False
        self.sampled_dynamic = True
        self._sampled_cap_lens = {}
        graph_enabled = (
            get_exec().graph.cuda_graph_config.decode.backend != Backend.DISABLED
        )
        for width in self.widths:
            if width == self.full:
                continue
            with spec_width(width):
                old_init = self.runner.init_new_workspace
                try:
                    backend = self.runner._get_attention_backend(
                        init_new_workspace=True
                    )
                finally:
                    self.runner.init_new_workspace = old_init
                graph = (
                    DecodeCudaGraphRunner(
                        self.runner,
                        attn_backend=backend,
                        speculative_num_steps=1,
                        speculative_num_draft_tokens=width,
                    )
                    if graph_enabled
                    else None
                )
            self.states[width] = (backend, graph)
            logger.info(
                "DSPine verify cap captured: draft=%d verify=%d graph=%s",
                self.full,
                width,
                graph is not None,
            )

    def begin(self, batch):
        from sglang.srt.speculative.dflash_worker_v2 import _is_all_greedy

        greedy = batch is None or _is_all_greedy(batch.sampling_info)
        self.sampled_round = not greedy
        sampled_eligible = (
            self.sampled_dynamic
            and batch is not None
            and (not getattr(batch.sampling_info, "is_any_greedy", True))
        )
        enabled = (
            batch is not None
            and batch.forward_mode.is_decode()
            and (greedy or sampled_eligible)
            and (not batch.has_grammar)
            and (not batch.return_logprob)
        )
        self.in_round = bool(enabled)
        self.executed_width = None
        bs = batch.batch_size() if enabled else 0
        self.active = self.policy.uniform_widths[bs - 1] if 1 <= bs <= 7 else self.full
        if enabled:
            self.counts[batch.batch_size(), self.active] += 1
            count = sum(self.counts.values())
            if count == 1 or count % 256 == 0:
                logger.info("DSPine actual verify widths: %s", dict(self.counts))

    def finish(self):
        self.restore_input()
        (self.runner.attn_backend, self.runner.decode_cuda_graph_runner) = self.states[
            self.full
        ]
        self.in_round = False
        self.active = self.full

    def prepare(self, original, verify_input, batch, target_worker, *args, **kwargs):
        if not self.in_round or self.active == self.full:
            return original(verify_input, batch, target_worker, *args, **kwargs)
        if (
            verify_input.ragged_verify_layout is not None
            or verify_input.custom_mask is not None
        ):
            raise ValueError(
                "Prefix verification requires ordinary causal chain verify"
            )
        (width, full, bs) = (self.active, self.full, batch.batch_size())
        names = (
            "draft_token",
            "positions",
            "draft_token_num",
            "num_tokens_per_req",
            "num_tokens_for_logprob_per_req",
            "kv_loc_cols",
        )
        self.saved_input = (
            verify_input,
            {n: getattr(verify_input, n) for n in names},
            batch,
            batch.out_cache_loc,
        )
        verify_input.draft_token = (
            verify_input.draft_token.view(bs, full)[:, :width].contiguous().view(-1)
        )
        verify_input.positions = (
            verify_input.positions.view(bs, full)[:, :width].contiguous().view(-1)
        )
        verify_input.draft_token_num = width
        verify_input.num_tokens_per_req = width
        verify_input.num_tokens_for_logprob_per_req = width
        verify_input.kv_loc_cols = slice(0, width)
        batch.out_cache_loc = (
            batch.out_cache_loc.view(bs, full)[:, :width].contiguous().view(-1)
        )
        (saved_cpu, saved_sum) = (batch.seq_lens_cpu, batch.seq_lens_sum)
        if saved_cpu is not None:
            batch.seq_lens_cpu = saved_cpu - (full - width)
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
        (self.runner.attn_backend, self.runner.decode_cuda_graph_runner) = self.states[
            width
        ]
        try:
            with spec_width(width):
                return original(verify_input, batch, target_worker, *args, **kwargs)
        finally:
            (batch.seq_lens_cpu, batch.seq_lens_sum) = (saved_cpu, saved_sum)

    def restore_input(self):
        if self.saved_input is None:
            return
        (vi, fields, batch, locations) = self.saved_input
        for name, value in fields.items():
            setattr(vi, name, value)
        batch.input_ids = fields["draft_token"]
        batch.out_cache_loc = locations
        self.saved_input = None

    def target(self, original, target_worker, *args, **kwargs):
        if (
            not self.in_round
            or self.active == self.full
            or (not kwargs.get("is_verify"))
        ):
            return original(target_worker, *args, **kwargs)
        try:
            with spec_width(self.active):
                result = original(target_worker, *args, **kwargs)
            result = copy.copy(result)
            result.logits_output = copy.copy(result.logits_output)
            output = result.logits_output
            (width, full) = (self.active, self.full)
            bs = self.saved_input[2].batch_size()
            actual_rows = int(output.next_token_logits.shape[0])
            if actual_rows != bs * width:
                raise RuntimeError(
                    f"Verify cap ineffective: expected {bs * width} rows, got {actual_rows}"
                )
            self.executed_width = width
            self.executed_counts[bs, width] += 1
            if self.executed_counts[bs, width] == 1:
                logger.info(
                    "DSPine verify executed: batch=%d draft=%d verify=%d actual_rows=%d cuda_graph=%s",
                    bs,
                    full,
                    width,
                    actual_rows,
                    result.can_run_cuda_graph,
                )
            logits = output.next_token_logits.view(bs, width, -1)
            from sglang_dspine.sampling.padding import logits_padding

            padded = logits_padding(
                self,
                logits,
                (bs, full, logits.shape[-1]),
            )
            padded[:, :width].copy_(logits)
            output.next_token_logits = padded.view(bs * full, -1)
            if output.hidden_states is not None:
                hidden = output.hidden_states.view(bs, width, -1)
                padded_hidden = hidden.new_zeros((bs, full, hidden.shape[-1]))
                padded_hidden[:, :width].copy_(hidden)
                output.hidden_states = padded_hidden.view(bs * full, -1)
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
        if self.executed_width != self.active:
            raise RuntimeError(
                "Capped acceptance requires a confirmed shortened target forward"
            )
        if self.sampled_round:
            from sglang_dspine.sampling.rejection import accept_sampled_prefix

            key = (bs, self.active)
            if key not in self._sampled_cap_lens:
                self._sampled_cap_lens[key] = torch.full(
                    (bs,), self.active, device=candidates.device, dtype=torch.int32
                )
            return accept_sampled_prefix(
                self,
                candidates=candidates,
                next_token_logits=next_token_logits,
                sampling_info=sampling_info,
                prefix_lens=prefix_lens,
                bs=bs,
                verify_lens=self._sampled_cap_lens[key],
            )
        top1 = torch.argmax(next_token_logits, dim=-1).view(bs, self.full)
        (accept, commit, bonus, output, new_seq) = (
            self.worker._next_accept_bonus_buffers(bs)
        )
        _accept_capped[bs,](
            candidates,
            top1,
            accept,
            commit,
            bonus,
            output,
            prefix_lens,
            new_seq,
            T=self.full,
            CAP=self.active,
            WIDTH=triton.next_power_of_2(self.full),
        )
        return (accept, commit, bonus, output, new_seq)


def active_controller():
    from sglang_dspine.runtime import worker as module

    current = module._active_worker
    return getattr(current, "verify_caps", None) if current is not None else None


def around_prepare(original, verify_input, batch, target_worker, *args, **kwargs):
    c = active_controller()
    if c is None or target_worker is not c.worker.target_worker:
        return original(verify_input, batch, target_worker, *args, **kwargs)
    return c.prepare(original, verify_input, batch, target_worker, *args, **kwargs)


def around_target(original, target_worker, *args, **kwargs):
    c = active_controller()
    if c is None or target_worker is not c.worker.target_worker:
        return original(target_worker, *args, **kwargs)
    return c.target(original, target_worker, *args, **kwargs)


def register_verify_cap_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.speculative.dflash_info.DFlashVerifyInput.prepare_for_verify",
        around_prepare,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.managers.tp_worker.TpModelWorker.forward_batch_generation",
        around_target,
        HookType.AROUND,
    )
