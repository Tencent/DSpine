"""SGLang MLX speculative worker for DSpine.

The worker is installed as SGLang's ``draft_worker`` (and therefore
``model_worker``) for ``--speculative-algorithm DSPINE``.  It reuses the MLX
target worker for prefill/extend, owns DSpine's per-request draft caches, and
runs one batched target verify per decode round.
"""

from __future__ import annotations

import dataclasses
import logging
from contextlib import nullcontext

import mlx.core as mx
import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker

from . import kernels
from .batched_verify import (
    greedy_verify_batched,
    sampled_verify_batched,
)
from .capture import CaptureSession, install_capture_layers
from .draft import DraftRuntime
from .kernels.filter import is_noop
from .plugin import set_active_worker
from .profiling import make_profiler_from_env
from .sampling import RequestSampling, seed_key, validate_request
from .spec import make_draft_input
from .verify import rollback_caches, run_batched_target_verify

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _ReqState:
    draft_caches: list
    sampling: RequestSampling
    prompt_len: int = 0
    rounds: int = 0
    accepted: int = 0


class DSpineMlxWorker(BaseSpecWorker):
    def __init__(self, server_args, gpu_id, ps, nccl_port, target_worker):
        super().__init__()
        self.server_args = server_args
        self._target_worker = target_worker
        self._draft_worker = None
        self._runner = target_worker._mlx_runner
        self._model = self._runner.model
        self._states: dict[str, _ReqState] = {}
        self.block_size = int(server_args.speculative_num_draft_tokens)
        self._prof = make_profiler_from_env()
        self._rng_key = mx.random.key(
            int(getattr(server_args, "random_seed", 0) or 0)
        )
        self._deterministic = _deterministic_inference()

        layout = self._runner._cache_layout
        if layout.has_auxiliary_state:
            raise NotImplementedError(
                "DSPINE MLX supports pure attention targets only "
                "(hybrid SSM/auxiliary state is not implemented)."
            )
        if layout.has_sliding_window_layers:
            raise NotImplementedError(
                "DSPINE MLX does not support sliding-window target layers."
            )
        from sglang.srt.hardware_backend.mlx.kv_cache.attention_wrapper import (
            MLXAttentionWrapper,
        )

        for idx in layout.attention_layer_indices:
            attn = getattr(layout.layers[idx], layout.attention_attr(idx))
            if not isinstance(attn, MLXAttentionWrapper):
                raise RuntimeError(
                    "DSPINE MLX expected every target attention layer to be "
                    f"wrapped by MLXAttentionWrapper; layer {idx} is "
                    f"{type(attn).__name__}."
                )

        self._draft = DraftRuntime.load(
            server_args.speculative_draft_model_path, self._model
        )
        self._draft.set_block_size(self.block_size)
        self._capture = CaptureSession(self._draft.cfg.target_layer_ids)
        self._capture_ids, self._capture_wrappers = install_capture_layers(
            self._model, self._draft.cfg.target_layer_ids
        )
        set_active_worker(self)
        kernels.warmup(block_size=self.block_size, k=self._draft.head_topk)
        logger.info(
            "DSPINE MLX worker ready: block=%d, target_layers=%s, "
            "draft_load=%.1fs, tables=%.1fs, kernels=%s",
            self.block_size,
            self._capture_ids,
            self._draft.load_seconds,
            self._draft.tables_seconds,
            kernels.get_config(),
        )

    # ------------------------------------------------------------------ API
    @property
    def draft(self) -> DraftRuntime:
        return self._draft

    @property
    def spec_v2_attn_backends(self):
        return ()

    @property
    def weight_load_time(self) -> float:
        return self._draft.load_seconds

    def _draft_model_runners(self):
        return ()

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        super().alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )
        self._target_worker._ensure_mlx_pool_initialized()

    def forward_batch_generation(
        self, batch, on_publish=None, grammar_barrier=None
    ) -> GenerationBatchResult:
        if on_publish is not None:
            raise RuntimeError(
                "DSPINE MLX does not support overlap scheduling "
                "(use --disable-overlap-schedule)."
            )
        if batch.forward_mode.is_idle():
            return self._idle_result(batch)

        self._reconcile(batch)
        for req in batch.reqs:
            # Admission already rejects unsupported features by setting
            # ``to_finish`` and queueing a one-token abort prefill; let that
            # request through so the scheduler can report the error.
            if (
                getattr(req, "to_finish", None) is not None
                or req.finished()
                or getattr(req, "is_retracted", False)
            ):
                continue
            error = validate_request(req, head_topk=self._draft.head_topk)
            if error is not None:
                raise RuntimeError(f"{error} (rid={req.rid})")

        if batch.forward_mode.is_extend():
            return self._forward_extend(batch)
        if batch.forward_mode.is_decode():
            return self._forward_decode(batch)
        raise RuntimeError(
            f"DSPINE MLX does not support forward mode {batch.forward_mode}"
        )

    # ------------------------------------------------------------- prefill
    def _forward_extend(self, batch) -> GenerationBatchResult:
        if getattr(batch, "decoding_reqs", None):
            raise RuntimeError(
                "DSPINE MLX does not support mixed decode inside an extend batch."
            )

        # A fresh prefill (extend start 0) must not inherit a stale cache from
        # a retracted request, or the MLX router treats it as a continuation.
        for req in batch.reqs:
            start = 0
            ext = getattr(req, "extend_range", None)
            if ext is not None:
                start = int(ext.start)
            if start == 0:
                self._states.pop(req.rid, None)
                if self._runner.has_request(req.rid):
                    self._runner.remove_request(req.rid)

        self._capture.begin(len(batch.reqs))
        try:
            out = self._target_worker.forward_batch_generation(batch)
            records = self._capture.end()
        except Exception:
            self._capture.abort_if_active()
            raise

        if len(records) != len(batch.reqs):
            raise RuntimeError(
                f"DSPINE MLX captured {len(records)} prefill records for "
                f"{len(batch.reqs)} requests"
            )
        if out.next_token_ids is None:
            raise RuntimeError("DSPINE MLX prefill produced no next token")

        for i, (req, record) in enumerate(zip(batch.reqs, records)):
            hidden = mx.concatenate(
                [record[lid] for lid in self._capture_ids], axis=-1
            )
            mx.eval(hidden)
            state = self._states.get(req.rid)
            if state is None:
                state = _ReqState(
                    draft_caches=self._draft.make_caches(),
                    sampling=RequestSampling.from_req(
                        req,
                        head_topk=self._draft.head_topk,
                        vocab_size=self._draft.vocab_size,
                        deterministic_seeding=self._deterministic,
                    ),
                )
                self._states[req.rid] = state
            self._draft.append_context(hidden, state.draft_caches)
            _eval_cache_states(state.draft_caches)
            state.prompt_len = int(batch.seq_lens[i])

        out.new_seq_lens = batch.seq_lens
        out.next_draft_input = make_draft_input(
            out.next_token_ids, batch.seq_lens, self.block_size
        )
        return out

    # -------------------------------------------------------------- decode
    def _forward_decode(self, batch) -> GenerationBatchResult:
        if not batch.reqs:
            return self._idle_result(batch)
        spec_info = batch.spec_info
        if spec_info is None or not hasattr(spec_info, "bonus_tokens"):
            raise RuntimeError(
                "DSPINE MLX decode requires a draft spec_info carrying bonus tokens."
            )
        prefix_lens = [int(x) for x in batch.seq_lens.tolist()]
        bonus = [int(x) for x in spec_info.bonus_tokens.tolist()]
        if len(prefix_lens) != len(batch.reqs) or len(bonus) != len(batch.reqs):
            raise RuntimeError(
                "DSPINE MLX decode metadata is not aligned with the batch"
            )

        caches = []
        draft_caches = []
        verify_keys = []
        anchors: list[int] = []
        samplings: list[RequestSampling] = []
        chain_keys: list[mx.array] = []
        with self._phase("proposal"):
            for i, req in enumerate(batch.reqs):
                state = self._states.get(req.rid)
                if state is None:
                    raise RuntimeError(
                        f"DSPINE MLX has no draft state for {req.rid}; it must be "
                        "prefilled first."
                    )
                cache = self._runner._req_caches.get(req.rid)
                if cache is None:
                    raise RuntimeError(f"MLX target cache missing for {req.rid}")
                offset = int(self._runner._first_attention_cache(cache).offset)
                if offset != prefix_lens[i]:
                    raise RuntimeError(
                        f"DSPINE MLX cache offset {offset} != seq_len "
                        f"{prefix_lens[i]} for {req.rid}"
                    )
                chain_key, verify_key = self._round_keys(state)
                anchors.append(int(bonus[i]))
                samplings.append(state.sampling)
                chain_keys.append(chain_key)
                verify_keys.append(verify_key)
                caches.append(cache)
                draft_caches.append(state.draft_caches)

            if kernels.dispatch.tail_batch_enabled():
                batched = self._draft.propose_batch(
                    draft_caches,
                    anchors,
                    self.block_size,
                    samplings,
                    chain_keys,
                    profiler=self._prof,
                )
                drafts = batched.drafts
                candidates = batched.candidates
                q_rows = batched.q_rows
                eval_args = [drafts, candidates]
                if q_rows is not None:
                    eval_args.append(q_rows)
                mx.async_eval(*eval_args)
            else:
                proposals = [
                    self._draft.propose(
                        draft_caches[i],
                        anchors[i],
                        self.block_size,
                        samplings[i],
                        chain_keys[i],
                    )
                    for i in range(len(batch.reqs))
                ]
                drafts = mx.stack([p.drafts for p in proposals])
                candidates = mx.stack([p.candidates for p in proposals])
                q_rows = (
                    None
                    if any(p.q_rows is None for p in proposals)
                    else mx.stack([p.q_rows for p in proposals])
                )
                _eval_proposals(proposals)
        verify_tokens = mx.concatenate(
            [
                mx.array(bonus, dtype=mx.int32)[:, None],
                drafts.astype(mx.int32),
            ],
            axis=1,
        )

        with self._phase("verify.forward"):
            logits, hidden = run_batched_target_verify(
                self._model,
                caches,
                prefix_lens,
                verify_tokens,
                capture=self._capture,
                num_layers=self._runner._cache_layout.num_layers,
                target_layer_ids=self._capture_ids,
            )
        if hidden is None:
            raise RuntimeError("DSPINE MLX verify did not capture target hidden")

        keep_lens: list[int] = []
        commit_lens: list[int] = []
        new_bonus: list[int] = []
        draft_lists: list[list[int]] = []
        with self._phase("verify.post"):
            accepts, bonuses = self._verify_batch(
                samplings, verify_keys, logits, drafts, candidates, q_rows
            )
            mx.eval(accepts, bonuses, drafts)
            accept_list = accepts.tolist()
            bonus_list = bonuses.tolist()
            draft_lists = drafts.tolist()
        commit_lens = [int(a) + 1 for a in accept_list]
        new_bonus = [int(b) for b in bonus_list]
        keep_lens = [
            prefix_lens[i] + commit_lens[i] for i in range(len(batch.reqs))
        ]

        with self._phase("rollback"):
            rollback_caches(caches, keep_lens)

        with self._phase("append_context"):
            for i, req in enumerate(batch.reqs):
                state = self._states[req.rid]
                accepted = commit_lens[i] - 1
                self._draft.append_context(
                    hidden[i : i + 1, : accepted + 1, :], state.draft_caches
                )
                _eval_cache_states(state.draft_caches)
                state.rounds += 1
                state.accepted += accepted

        with self._phase("output.pack"):
            T = self.block_size
            out_tokens = torch.zeros((len(batch.reqs), T), dtype=torch.long)
            for i, drafts in enumerate(draft_lists):
                accepted = commit_lens[i] - 1
                if accepted:
                    out_tokens[i, :accepted] = torch.tensor(
                        drafts[:accepted], dtype=torch.long
                    )
                out_tokens[i, accepted] = int(new_bonus[i])

        next_token_ids = out_tokens.reshape(-1)
        accept_lens = torch.tensor(commit_lens, dtype=torch.int32)
        new_seq_lens = torch.tensor(keep_lens, dtype=torch.int64)
        if self._prof is not None:
            self._prof.round_done(
                requests=len(batch.reqs),
                block=self.block_size,
                accepted=sum(commit_lens),
                accept_lens=commit_lens,
            )
        return GenerationBatchResult(
            logits_output=LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=next_token_ids,
            accept_lens=accept_lens,
            new_seq_lens=new_seq_lens,
            next_draft_input=make_draft_input(
                torch.tensor(new_bonus, dtype=torch.long),
                new_seq_lens,
                T,
            ),
            speculative_num_draft_tokens=T,
            can_run_cuda_graph=False,
        )

    # ------------------------------------------------------------- helpers
    def _phase(self, name: str):
        if self._prof is None:
            return nullcontext()
        return self._prof.phase(name)

    def profile_snapshot(self):
        """Last N rounds of phase timings (``None`` unless profiling is on)."""
        return None if self._prof is None else self._prof.snapshot()

    def _verify_batch(
        self, samplings, verify_keys, logits, drafts, candidates, q_rows
    ) -> tuple[mx.array, mx.array]:
        """One batched greedy/sampled verification for the whole decode batch."""
        greedy_flags = [bool(s.greedy) for s in samplings]
        if all(greedy_flags):
            return greedy_verify_batched(logits, drafts)
        if candidates is None or q_rows is None:
            raise RuntimeError("sampled request has no refinement q rows")

        n = int(drafts.shape[1])
        splits = [mx.random.split(k, 3) for k in verify_keys]
        u_accept = mx.stack(
            [mx.random.uniform(key=s[0], shape=(n,)) for s in splits]
        )
        u_final = mx.stack([mx.random.uniform(key=s[1]) for s in splits])
        u_res = mx.stack([mx.random.uniform(key=s[2]) for s in splits])
        inv = mx.array([float(s.inv_temp) for s in samplings], dtype=mx.float32)
        topk = mx.array(
            [int(s.target_top_k) for s in samplings], dtype=mx.int32
        )
        topp = mx.array([float(s.top_p) for s in samplings], dtype=mx.float32)
        minp = mx.array([float(s.min_p) for s in samplings], dtype=mx.float32)
        vocab = int(logits.shape[-1])
        skip_noop = all(
            is_noop(s.target_top_k, s.top_p, s.min_p, vocab) for s in samplings
        )
        apply_min_p = any(float(s.min_p) > 0.0 for s in samplings)

        s_accept, s_bonus = sampled_verify_batched(
            logits,
            drafts,
            candidates,
            q_rows,
            inv_temp=inv,
            top_k=topk,
            top_p=topp,
            min_p=minp,
            u_accept=u_accept,
            u_final=u_final,
            u_res=u_res,
            skip_noop=skip_noop,
            apply_min_p=apply_min_p,
        )
        if not any(greedy_flags):
            return s_accept, s_bonus
        g_accept, g_bonus = greedy_verify_batched(logits, drafts)
        mask = mx.array(greedy_flags)
        return (
            mx.where(mask, g_accept, s_accept),
            mx.where(mask, g_bonus, s_bonus),
        )

    def _idle_result(self, batch) -> GenerationBatchResult:
        empty_long = torch.empty((0,), dtype=torch.long)
        empty_lens = torch.empty((0,), dtype=torch.int64)
        return GenerationBatchResult(
            logits_output=LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=empty_long,
            accept_lens=torch.empty((0,), dtype=torch.int32),
            new_seq_lens=empty_lens,
            next_draft_input=make_draft_input(
                empty_long, empty_lens, self.block_size
            ),
            speculative_num_draft_tokens=self.block_size,
            can_run_cuda_graph=False,
        )

    def _round_keys(self, state: _ReqState):
        if state.sampling.seed is not None:
            key = seed_key(state.sampling.seed, state.rounds)
        else:
            self._rng_key, key = mx.random.split(self._rng_key)
        return mx.random.split(key, 2)

    def _reconcile(self, batch) -> None:
        # A prefill/extend batch may run while other requests stay in the
        # running decode batch; their draft state must survive it.  Only a
        # decode batch is a complete snapshot of the running set.
        if not batch.forward_mode.is_decode():
            return
        current = {req.rid for req in batch.reqs}
        for rid in list(self._runner._req_caches):
            if rid not in current:
                self._runner.remove_request(rid)
        for rid in list(self._states):
            if rid not in current:
                self._states.pop(rid, None)

    # --------------------------------------------------------- lifecycle
    def note_request_finished(self, *, rid: str, natural_stop: bool) -> None:
        self._states.pop(rid, None)

    def prepare_for_kv_cache_release(self, req) -> None:
        try:
            self._target_worker.prepare_for_kv_cache_release(req)
        finally:
            self._states.pop(req.rid, None)

    def on_request_released(self, rid) -> None:
        if rid is None:
            return
        self._states.pop(rid, None)
        if self._runner.has_request(rid):
            self._runner.remove_request(rid)

    def clear_cache_pool(self):
        self._states.clear()
        mx.clear_cache()

    def update_weights_from_disk(self, recv_req):
        raise NotImplementedError("DSPINE MLX does not support weight updates.")

    def update_weights_from_ipc(self, recv_req):
        raise NotImplementedError("DSPINE MLX does not support weight updates.")


def _eval_cache_states(caches) -> None:
    arrays = []
    for cache in caches:
        keys = getattr(cache, "keys", None)
        values = getattr(cache, "values", None)
        if keys is not None:
            arrays.append(keys)
        if values is not None:
            arrays.append(values)
    if arrays:
        mx.async_eval(*arrays)


def _eval_proposals(proposals) -> None:
    arrays = []
    for proposal in proposals:
        arrays.append(proposal.drafts)
        arrays.append(proposal.candidates)
        if proposal.q_rows is not None:
            arrays.append(proposal.q_rows)
    if arrays:
        mx.eval(*arrays)


def _deterministic_inference() -> bool:
    try:
        from sglang.srt.runtime_context import get_exec

        return bool(get_exec().deterministic.enable_deterministic_inference)
    except Exception:
        return False
