"""DSpine serving on SGLang 0.5.22."""

import logging
import os
from typing import Optional, Tuple

import torch
from sglang.kernels.ops.speculative.dspark.dspark_accept import (
    SoftmaxTemp,
)
from sglang.kernels.ops.speculative.reject_sampling import (
    chain_speculative_sampling_triton,
)
from sglang.srt.lora.layers import BaseLayerWithLoRA
from sglang.srt.runtime_context import get_exec, get_parallel, get_schedule
from sglang.srt.speculative.dflash_utils import (
    _dflash_top_k_renorm_prob,
    _dflash_top_p_renorm_prob,
    _get_or_create_chain_verify_buffers,
    is_dflash_sampling_verify_available,
)
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

from sglang_dspine.kernels import dspine as dspine_kernels
from sglang_dspine.models.dspine import DSpineDraftModel
from sglang_dspine.sampling.random import (
    DRAFT,
    RequestRNG,
    seeded_batch,
    verification_uniforms,
)

logger = logging.getLogger(__name__)
SAMPLED_VERIFY_TARGET = "sglang.srt.speculative.dflash_worker_v2.compute_dflash_sampling_correct_drafts_and_bonus"
_active_worker: Optional["DSpineWorkerV2"] = None
SAMPLED_POLICIES = ("sampler",)


def _is_all_greedy(sampling_info) -> bool:
    return sampling_info is None or sampling_info.is_all_greedy


def unwrap_lora_layer(module):
    return module.base_layer if isinstance(module, BaseLayerWithLoRA) else module


def _torch_backend_filtered_probs(probs, top_ks, top_ps, min_ps, need_min_p):
    """The filtered weights of SGLang's pytorch sampling backend
    (layers/sampler.py:top_k_top_p_min_p_sampling_from_probs_torch): probs sorted descending, ranks
    >= top_k zeroed, then rows whose preceding mass exceeds top_p, then (min-p on) rows below
    min_p * the top prob.  Returns (filtered sorted probs, their token ids)."""
    (probs_sort, probs_idx) = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[
        torch.arange(0, probs.shape[-1], device=probs.device).view(1, -1)
        >= top_ks.view(-1, 1)
    ] = 0.0
    probs_sort[probs_sum - probs_sort > top_ps.view(-1, 1)] = 0.0
    if need_min_p:
        min_p_thresholds = probs_sort[:, 0] * min_ps
        probs_sort[probs_sort < min_p_thresholds.view(-1, 1)] = 0.0
    return (probs_sort, probs_idx.to(torch.int32))


class DSpineDraftSampler:
    """The DSpine refinement tail over one draft forward.

    ``__call__`` is capture-safe and is folded into the draft CUDA graph (candidates, the greedy
    chain, copies of the refinement state into static buffers); ``sample`` runs eagerly after it,
    only on rounds with a sampled request.  ``stage_sampling_params`` opens a round; its sampled
    chain runs after the round's draft forward, from ``after_graph_replay`` or from the worker's
    eager path, and ``round_sampled`` tells verification that it did.  Owns every buffer either
    side reads or writes: staged sampling params (inv_temp, top_k, top_p, min_p, greedy), the
    copied state (raw, rms, query, anchors), uniforms, and the outputs greedy_out / out
    [max_bs*(T-1)], candidate_out / q_out [max_bs, T-1, K].
    """

    def __init__(self, *, draft_model, lm_head_weight, block_size, max_bs, device):
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        if torch.get_float32_matmul_precision() != "highest" or bool(
            torch.backends.cuda.matmul.allow_tf32
        ):
            raise RuntimeError(
                "DSPINE computes its sampling table in true fp32: TF32 matmuls must be off (do not enable TF32 for this server)"
            )
        bridge = draft_model.selected_prefix_bridge
        if not bridge._ready:
            raise RuntimeError(
                "DSPINE sampler built before the draft weights were prepared"
            )
        self.policy = "sampler"
        self.draft_model = draft_model
        self.lm_head_weight = lm_head_weight
        self.block_size = int(block_size)
        self.k = int(draft_model.head_topk)
        self.max_bs = int(max_bs)
        t = self.block_size
        gamma = t - 1
        n = self.max_bs * gamma
        k = self.k
        h = int(lm_head_weight.shape[1])
        r = int(bridge.rank)
        last = len(draft_model.layers) - 1
        f32 = torch.float32
        (self.tg, self.tsv) = dspine_kernels.build_serial_tables(
            lm_head_weight,
            draft_model.norm.weight.detach(),
            bridge.feature_write_weights[last].detach(),
            bridge.packed_weight[r:].detach(),
            bridge.codebook,
        )
        bridge.fold_terminal_sender_table(self.tsv)
        if bridge.terminal_residual_rank:
            self.res_emb = bridge.terminal_residual_embedding
            self.res_proj_f32 = (
                bridge.terminal_residual_projection.detach().to(f32).contiguous()
            )
        else:
            self.res_emb = self.res_proj_f32 = None
        self.pred_gate_f32 = (
            bridge.predecessor_gate.weight.detach().to(f32).contiguous()
        )
        self.prev_f32 = bridge.prev.weight.detach().to(f32).contiguous()
        self.gate_b_f32 = bridge.gate.bias.detach().to(f32).contiguous()
        self.write_f32 = (
            bridge.feature_write_weights[last].detach().to(f32).contiguous()
        )
        self.norm_f32 = draft_model.norm.weight.detach().to(f32).contiguous()
        self.norm_eps = float(draft_model.norm.variance_epsilon)
        self.inv_temp = torch.ones((self.max_bs,), dtype=f32, device=device)
        self.top_k = torch.full((self.max_bs,), k, dtype=torch.int32, device=device)
        self.top_p = torch.ones((self.max_bs,), dtype=f32, device=device)
        self.min_p = torch.zeros((self.max_bs,), dtype=f32, device=device)
        self.greedy = torch.ones((self.max_bs,), dtype=torch.int32, device=device)
        self.raw = torch.zeros(
            (self.max_bs * t, h), dtype=torch.bfloat16, device=device
        )
        self.rms = torch.ones((self.max_bs * t,), dtype=f32, device=device)
        self.query = torch.zeros(
            (self.max_bs * t, r), dtype=torch.bfloat16, device=device
        )
        self.anchors = torch.zeros((self.max_bs,), dtype=torch.int64, device=device)
        self.uniforms = torch.zeros((n,), dtype=f32, device=device)
        self.greedy_out = torch.zeros((n,), dtype=torch.int64, device=device)
        self.out = torch.zeros((n,), dtype=torch.int64, device=device)
        self.candidate_out = torch.zeros(
            (self.max_bs, gamma, k), dtype=torch.int64, device=device
        )
        self.q_out = torch.zeros((self.max_bs, gamma, k), dtype=f32, device=device)
        self.optimized = (
            tuple(lm_head_weight.shape) == (248320, 5120)
            and t == 16
            and k == 16
            and r == 512
        )
        self.head = None
        self.proposal_cache = None
        self.confidence_enabled = self.optimized
        self.confidence_table = (
            torch.empty((n, k), dtype=f32, device=device)
            if self.confidence_enabled
            else None
        )
        self.confidence_out = (
            torch.empty((self.max_bs, gamma), dtype=f32, device=device)
            if self.confidence_enabled
            else None
        )
        self.random = RequestRNG()
        self._round_bs = 0
        self._round_samples = False
        self._round_sampled = False

    @property
    def round_bs(self) -> int:
        """Batch size of the open round (0 before the first stage)."""
        return self._round_bs

    @property
    def round_samples(self) -> bool:
        """The open round has a sampled request (its sampled chain must run)."""
        return self._round_samples

    @property
    def round_sampled(self) -> bool:
        """The open round's sampled chain has run (out / q_out hold it)."""
        return self._round_sampled

    def stage_sampling_params(self, *, bs: int, sampling_info) -> None:
        """Open a round before its draft forward: refresh the static sampling params (device
        copies, no host sync).  Greedy = top_k <= 1, SGLang's own rule (temperature 0 is
        normalised to top_k 1)."""
        if bs > self.max_bs:
            raise RuntimeError(
                f"DSPINE batch {bs} exceeds the sampler capacity {self.max_bs}"
            )
        self._round_bs = int(bs)
        self._round_samples = not _is_all_greedy(sampling_info)
        self._round_sampled = False
        if self.head is not None:
            self.head.actual_bs.fill_(bs)
        if not self._round_samples:
            self.greedy[:bs].fill_(1)
            return
        top_ks = sampling_info.top_ks.view(-1)[:bs]
        torch.reciprocal(
            torch.clamp(
                sampling_info.temperatures.view(-1)[:bs].to(torch.float32), min=1e-05
            ),
            out=self.inv_temp[:bs],
        )
        self.top_k[:bs].copy_(torch.clamp(top_ks, min=1, max=self.k).to(torch.int32))
        self.top_p[:bs].copy_(sampling_info.top_ps.view(-1)[:bs].to(torch.float32))
        self.min_p[:bs].copy_(sampling_info.min_ps.view(-1)[:bs].to(torch.float32))
        self.greedy[:bs].copy_((top_ks <= 1).to(torch.int32))

    @torch.no_grad()
    def __call__(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> None:
        """Capture-safe part: candidates, the greedy chain, and the state copy.  hidden_states
        [bs*T, H] is the draft forward's output; input_ids [bs*T] the block ids (anchor first)."""
        (m, h) = hidden_states.shape
        t = self.block_size
        if m % t:
            raise RuntimeError("DSPINE tail expects whole request blocks")
        bs = m // t
        self.propose(hidden_states.view(bs, t, h)[:, 1:, :].reshape(-1, h), input_ids)

    @torch.no_grad()
    def propose(self, rows: torch.Tensor, input_ids: torch.Tensor) -> None:
        """The tail on the proposal rows: rows [bs*(T-1), H] are rows 1..T-1 of each block of the
        draft forward's output, input_ids [bs*T] the block ids (only column 0, the anchor, is
        read)."""
        dm = self.draft_model
        stash = dm.last_write_stash
        if stash is None:
            raise RuntimeError(
                "DSPINE tail ran without the draft forward's last-write stash"
            )
        bridge = dm.selected_prefix_bridge
        (t, k, r) = (self.block_size, self.k, int(bridge.rank))
        (n, h) = rows.shape
        if n % (t - 1) or input_ids.numel() != n // (t - 1) * t:
            raise RuntimeError("DSPINE tail expects whole request blocks")
        bs = n // (t - 1)
        m = bs * t
        if bs > self.max_bs:
            raise RuntimeError(
                f"DSPINE batch {bs} exceeds the sampler capacity {self.max_bs}"
            )
        if self.head is None:
            logits = torch.matmul(rows, self.lm_head_weight.T)
            cand = dspine_kernels.topk_ids(logits, k)
            del logits
        else:
            cand = self.head.retrieve(rows)
        idx = dspine_kernels.serial_table(
            stash["raw"],
            dm.norm.weight,
            cand,
            input_ids,
            self.lm_head_weight,
            self.tg,
            self.tsv,
            stash["packed"],
            stash["rms"],
            bridge.gate.bias,
            t,
            confidence=self.confidence_table[:n] if self.confidence_enabled else None,
        )
        dspine_kernels.serial_chain(
            idx,
            cand,
            self.greedy_out,
            t,
            confidence=self.confidence_table[:n] if self.confidence_enabled else None,
            confidence_out=self.confidence_out[:bs].view(-1)
            if self.confidence_enabled
            else None,
        )
        self.out[:n].copy_(self.greedy_out[:n])
        self.candidate_out[:bs].copy_(cand.view(bs, t - 1, k))
        self.raw[:m].copy_(stash["raw"])
        self.rms[:m].copy_(stash["rms"])
        self.query[:m].copy_(stash["packed"][:, :r])
        self.anchors[:bs].copy_(input_ids.view(bs, t)[:, 0])

    @torch.no_grad()
    def sample(self, bs: int) -> None:
        """Eager sampled chain for the round just drafted by ``__call__`` (same batch)."""
        if bs > self.max_bs:
            raise RuntimeError(
                f"DSPINE batch {bs} exceeds the sampler capacity {self.max_bs}"
            )
        dm = self.draft_model
        bridge = dm.selected_prefix_bridge
        (t, k) = (self.block_size, self.k)
        (m, n) = (bs * t, bs * (t - 1))
        cand = self.candidate_out[:bs].view(n, k)
        if self.proposal_cache is not None:
            table = self.proposal_cache.lookup(bs)
            if table is None:
                raise RuntimeError(
                    "DSpine proposal weights, graph buffers or execution stream changed"
                )
        else:
            table = dspine_kernels.serial_rs_table(
                self.raw[:m],
                self.rms[:m],
                self.query[:m],
                self.anchors[:bs],
                cand,
                bridge.codebook,
                self.pred_gate_f32,
                self.prev_f32,
                self.gate_b_f32,
                self.write_f32,
                self.norm_f32,
                self.lm_head_weight,
                self.norm_eps,
                t,
                res_emb=self.res_emb,
                res_proj=self.res_proj_f32,
            )
        if self.random.seeds is None:
            self.uniforms[:n].uniform_()
        else:
            self.uniforms[:n].copy_(self.random.draw(t - 1, DRAFT).reshape(-1))
        dspine_kernels.serial_sample_chain(
            table,
            self.inv_temp,
            self.top_k,
            self.top_p,
            self.min_p,
            self.greedy,
            self.uniforms,
            cand,
            self.greedy_out,
            self.out,
            self.q_out.view(-1, k),
            t,
        )
        self._round_sampled = True

    def after_graph_replay(self, forward_batch) -> None:
        """Run by DSpineDraftCudaGraphRunner after every draft graph replay, before the worker
        reads ``out``.  The replay has just written the greedy chain to ``out``; a round with a
        sampled request replaces it with the chain sampled from the replayed state."""
        if not self._round_samples:
            return
        if int(forward_batch.batch_size) != self._round_bs:
            raise RuntimeError(
                f"DSPINE draft replay for batch {int(forward_batch.batch_size)} while the staged round has batch {self._round_bs}"
            )
        self.sample(self._round_bs)


def build_dspine_target_probs(
    *, next_token_logits, sampling_info, block_size: int, bs: int, sampling_backend: str
) -> torch.Tensor:
    """[bs, T, V] fp32: the distribution SGLang's ordinary Sampler (srt/layers/sampler.py,
    Sampler._sample_from_probs) draws from for each verify row: softmax(logits / temperature),
    then the configured backend's filters, renormalised:
      flashinfer, min-p off: joint top-k / top-p (the sampler's filter_apply_order="joint"): the
          support is the top_k_renorm_prob support AND the top_p_renorm_prob support, both taken
          on the unfiltered probs -- the same primitives the sampler's own mask capture uses;
      flashinfer, min-p on: top_k_renorm_prob -> top_p_renorm_prob -> keep p >= min_p * max p
          (the sampler's sequential min-p branch);
      pytorch: the filtered weights of top_k_top_p_min_p_sampling_from_probs_torch
          (_torch_backend_filtered_probs).
    Rejection sampling against this p makes the committed tokens follow non-speculative sampling."""
    t = int(block_size)
    probs = SoftmaxTemp.execute(
        logits=next_token_logits,
        temperatures=sampling_info.temperatures,
        rows_per_request=t,
    ).to(torch.float32)
    return filter_target_probs(probs, sampling_info, t, bs, sampling_backend)


def filter_target_probs(probs, sampling_info, t, bs, sampling_backend, normalize=True):
    need_k = bool(sampling_info.need_top_k_sampling)
    need_p = bool(sampling_info.need_top_p_sampling)
    need_min_p = bool(sampling_info.need_min_p_sampling)
    if not (need_k or need_p or need_min_p):
        return probs.view(bs, t, -1)
    top_ks = torch.repeat_interleave(sampling_info.top_ks.view(-1), t)
    top_ps = torch.repeat_interleave(sampling_info.top_ps.view(-1), t)
    min_ps = torch.repeat_interleave(sampling_info.min_ps.view(-1), t)
    if sampling_backend == "pytorch":
        (weights, token_ids) = _torch_backend_filtered_probs(
            probs, top_ks, top_ps, min_ps, need_min_p
        )
        probs = torch.zeros_like(probs).scatter_(-1, token_ids.long(), weights)
    elif sampling_backend == "flashinfer":
        if need_min_p:
            probs = _dflash_top_k_renorm_prob(probs, top_ks)
            probs = _dflash_top_p_renorm_prob(probs, top_ps)
            probs = probs.masked_fill(
                probs < probs.amax(dim=-1, keepdim=True) * min_ps.view(-1, 1), 0.0
            )
        else:
            keep = None
            if need_k:
                keep = _dflash_top_k_renorm_prob(probs, top_ks) > 0
            if need_p:
                in_nucleus = _dflash_top_p_renorm_prob(probs, top_ps) > 0
                keep = in_nucleus if keep is None else keep.logical_and_(in_nucleus)
            probs = probs.masked_fill(~keep, 0.0)
    else:
        raise ValueError(
            f"DSPINE sampled verify: unsupported sampling backend {sampling_backend!r}"
        )
    if normalize:
        probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs.view(bs, t, -1)


def dspine_accept_sampling(
    *,
    target_probs=None,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    candidate_ids: torch.Tensor,
    q_rows: torch.Tensor,
    sampling_info,
    sampling_backend: str,
    dense_q: torch.Tensor,
    random_state=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chain rejection sampling of the drafted block (module docstring, step 5).

    candidates [bs, T] = [anchor, T-1 drafts]; candidate_ids / q_rows [bs, T-1, K] give the sparse
    q each draft was sampled from (draft i = candidates[:, i+1] belongs to row i); dense_q is a
    zeroed [>=bs, T-1, V] fp32 buffer, scattered into and zeroed back here.  Returns
    (accept_len int32 [bs], bonus int64 [bs])."""
    (bs, block) = candidates.shape
    gamma = block - 1
    device = candidates.device
    if (
        tuple(candidate_ids.shape[:2]) != (bs, gamma)
        or candidate_ids.shape != q_rows.shape
    ):
        raise RuntimeError("DSPINE verify: candidate / q layout mismatch")
    if target_probs is None:
        target_probs = build_dspine_target_probs(
            next_token_logits=target_logits,
            sampling_info=sampling_info,
            block_size=block,
            bs=bs,
            sampling_backend=sampling_backend,
        )
    (
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        predicts,
        accept_index,
        accept_token_num,
    ) = _get_or_create_chain_verify_buffers(bs=bs, draft_token_num=block, device=device)
    uniform_samples, uniform_final = verification_uniforms(
        bs, block, device, random_state
    )
    draft_probs = dense_q[:bs]
    try:
        draft_probs.scatter_(-1, candidate_ids, q_rows.to(torch.float32))
        chain_speculative_sampling_triton(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            retrive_index=retrieve_index,
            retrive_next_token=retrieve_next_token,
            retrive_next_sibling=retrieve_next_sibling,
            uniform_samples=uniform_samples,
            uniform_samples_for_final_sampling=uniform_final,
            target_probs=target_probs,
            draft_probs=draft_probs,
            threshold_single=1.0,
            threshold_acc=1.0,
            deterministic=True,
        )
    finally:
        draft_probs.scatter_(-1, candidate_ids, 0.0)
    accept_len = accept_token_num.clone()
    rows = torch.arange(bs, dtype=torch.int64, device=device)
    accept_pos = accept_index[rows, accept_len.to(torch.int64)].to(torch.int64)
    bonus = predicts[accept_pos].to(torch.int64)
    return (accept_len.to(torch.int32), bonus)


class DSpineWorkerV2(DFlashWorkerV2):
    """DFlashWorkerV2 with DSpine's sampler on SGLang 0.5.22's seams A-C (module docstring)."""

    def __init__(self, server_args, gpu_id: int, nccl_port: int, target_worker):
        super().__init__(server_args, gpu_id, nccl_port, target_worker)
        dm = self.draft_model
        if not isinstance(dm, DSpineDraftModel):
            raise ValueError(
                f"--speculative-algorithm DSPINE loaded draft class {type(dm).__name__}; expected DSpineDraftModel"
            )
        if int(get_parallel().tp_group.world_size) != 1:
            raise ValueError("DSPINE supports tp=1 only")
        if self.use_compact_draft_cache:
            raise ValueError("DSPINE does not support --speculative-draft-window-size")
        dm.set_block_size(int(self.block_size))
        if dm.native_swa_window is not None:
            from sglang_dspine.layers.attention import validate_native_swa_runner

            validate_native_swa_runner(
                self.draft_model_runner, served_block_size=int(self.block_size)
            )
        target_model = self._target_worker.model_runner.model
        lm_head = unwrap_lora_layer(getattr(target_model, "lm_head", None))
        weight = getattr(lm_head, "weight", None)
        if (
            weight is None
            or weight.dtype != torch.bfloat16
            or weight.dim() != 2
            or (weight.stride(1) != 1)
        ):
            raise ValueError(
                "DSPINE needs an unquantized row-contiguous BF16 target lm_head"
            )
        num_org = int(weight.shape[0])
        shard = getattr(lm_head, "shard_indices", None)
        if shard is not None:
            if int(shard.num_added_elements) or int(shard.org_vocab_start_index):
                raise ValueError("DSPINE: added or offset target vocabulary")
            num_org = int(shard.num_org_elements)
        bridge = dm.selected_prefix_bridge
        if num_org != int(bridge.codebook.shape[0]):
            raise ValueError(
                f"DSPINE: target vocabulary {num_org} != code-bridge codebook {int(bridge.codebook.shape[0])}"
            )
        if int(weight.shape[1]) != int(dm.config.hidden_size):
            raise ValueError("DSPINE: target lm_head width != draft hidden size")
        graph_bs = list(get_exec().graph.cuda_graph_config.decode.bs or [])
        max_bs = max(graph_bs + [int(get_schedule().max_running_requests or 0), 1])
        self.sampled_policy = "sampler"
        self._dspine_sampler = DSpineDraftSampler(
            draft_model=dm,
            lm_head_weight=weight[:num_org],
            block_size=self.block_size,
            max_bs=max_bs,
            device=self.device,
        )
        self._dspine_dense_q: Optional[torch.Tensor] = None
        self._warned_min_p = False
        self._warned_dense_q_realloc = False
        self.dspine_round_counts = {"staged": 0, "sampled": 0, "eager": 0}
        dm.dspine_sampler = self._dspine_sampler
        self.target_distribution = None
        if self._dspine_sampler.optimized:
            if max_bs > 8:
                raise ValueError(
                    "Qwen3.8-27B DSpine supports at most eight concurrent requests"
                )
            from sglang_dspine.sampling.target import TargetDistribution

            self.target_distribution = TargetDistribution(self)
        self._check_seams(server_args)
        global _active_worker
        if _active_worker is not None and _active_worker is not self:
            raise RuntimeError("DSPINE runs one draft worker per process (tp=1)")
        _active_worker = self
        if self._target_tp_rank == 0:
            conv = dm.layers[0].attention_conv
            logger.info(
                "DSPINE ready: block_size=%d head_topk=%d bridge=%s terminal_residual_rank=%d conv=%s sampled_policy=%s trained_block_size=%d sampler_capacity_bs=%d verify: greedy=argmax chain, sampled=chain rejection sampling over the sampled serial chain",
                self.block_size,
                self._dspine_sampler.k,
                bridge.implementation,
                bridge.terminal_residual_rank,
                f"taps{conv.taps}/group{conv.group_size}"
                if conv is not None
                else "none",
                self.sampled_policy,
                dm.trained_block_size,
                max_bs,
            )

    def _update_target_mamba_state_after_verify(
        self, *, batch, seq_lens_pre_verify, seq_lens_post_verify, commit_lens
    ):
        from sglang_dspine.runtime.replayssm.state import commit_gdn_replay

        if commit_gdn_replay(
            self,
            batch=batch,
            commit_lens=commit_lens,
            seq_lens_pre_verify=seq_lens_pre_verify,
            seq_lens_post_verify=seq_lens_post_verify,
        ):
            return
        return super()._update_target_mamba_state_after_verify(
            batch=batch,
            seq_lens_pre_verify=seq_lens_pre_verify,
            seq_lens_post_verify=seq_lens_post_verify,
            commit_lens=commit_lens,
        )

    def forward_batch_generation(self, *args, **kwargs):
        controller = getattr(self, "verify_caps", None)
        batch = kwargs.get("batch", args[0] if args else None)
        if (
            controller is None
            and self._dspine_sampler.optimized
            and (batch is not None)
            and batch.forward_mode.is_decode()
        ):
            raise RuntimeError("Requested DSPine verify caps were not initialized")
        with seeded_batch(batch):
            if controller is None:
                return super().forward_batch_generation(*args, **kwargs)
            controller.begin(batch)
            try:
                return super().forward_batch_generation(*args, **kwargs)
            finally:
                controller.finish()

    def _accept_block(self, **kwargs):
        controller = getattr(self, "verify_caps", None)
        if (
            controller is not None
            and controller.in_round
            and (controller.active < controller.full)
        ):
            result = controller.accept(**kwargs)
        else:
            result = super()._accept_block(**kwargs)
        if controller is not None and hasattr(controller, "observe"):
            controller.observe(result[0])
        return result

    def _measure_sampling_peak(self, *, candidates, target_logits, sampling_info):
        """Rehearse DSPine's own rejection sampler before any live draft chain exists.

        SGLang 0.5.22 calls this at startup to size graph-pool borrowing. A valid
        synthetic one-hot proposal tests the real verifier without stale round state.
        """
        (bs, block) = candidates.shape
        gamma = block - 1
        k = self._dspine_sampler.k
        candidate_ids = torch.arange(k, dtype=torch.int64, device=candidates.device)
        candidate_ids = candidate_ids.view(1, 1, k).expand(bs, gamma, k).contiguous()
        q_rows = torch.zeros(
            (bs, gamma, k), dtype=torch.float32, device=candidates.device
        )
        q_rows[..., 0] = 1.0
        proposals = torch.zeros_like(candidates)
        dense_q = self._dense_q_buffer(bs, int(target_logits.shape[-1]))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base_bytes = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        self._dspine_verify(
            candidates=proposals,
            next_token_logits=target_logits,
            candidate_ids=candidate_ids,
            q_rows=q_rows,
            sampling_info=sampling_info,
        )
        torch.cuda.synchronize()
        return torch.cuda.memory_stats()["allocated_bytes.all.peak"] - base_bytes

    def _check_seams(self, server_args) -> None:
        """Fail at startup where SGLang 0.5.22 would otherwise serve DSpine silently wrong."""
        if not is_dflash_sampling_verify_available():
            raise RuntimeError(
                "DSPINE requires SGLang's sampled verification kernels (sgl_kernel)"
            )
        import sglang.srt.speculative.dflash_worker_v2 as dflash_worker_v2
        from sglang.srt.plugins.hook_registry import HookRegistry

        verify = dflash_worker_v2.compute_dflash_sampling_correct_drafts_and_bonus
        if (
            SAMPLED_VERIFY_TARGET not in HookRegistry._patched
            or getattr(verify, "__wrapped__", None) is None
        ):
            raise RuntimeError(
                "DSPINE's sampled-verify hook is not installed: load the plugin through SGLang's plugin loader (pip install sglang-dspine)"
            )
        path = getattr(server_args, "speculative_draft_model_path", None)
        if path and os.path.isfile(os.path.join(path, "mask_embedding.pt")):
            raise ValueError(
                "DSPINE drafts carry no mask_embedding.pt (SGLang 0.5.22 would not merge it)"
            )

    def init_attention_backends(self):
        super().init_attention_backends()
        if self.draft_model.native_swa_window is not None:
            from sglang.srt.layers.attention.flashattention_backend import (
                FlashAttentionBackend,
            )

            runner = self.draft_model_runner
            backend = runner.attn_backend
            if runner.draft_attention_backend in ("fa3", "fa4") and (
                type(backend) is not FlashAttentionBackend
                or backend.fa_impl_ver != int(runner.draft_attention_backend[-1])
            ):
                raise ValueError(
                    "Native DSPINE SWA candidate requires the resolved FA3/FA4 backend itself"
                )

    def alloc_memory_pool(self, *args, **kwargs):
        result = super().alloc_memory_pool(*args, **kwargs)
        if self.draft_model.native_swa_window is not None:
            from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

            pool = self.draft_model_runner.token_to_kv_pool
            if type(pool) is not MHATokenToKVPool or pool.dtype != torch.bfloat16:
                raise ValueError(
                    "Native DSPINE SWA candidate requires the standard dense BF16 MHATokenToKVPool; bounded/hybrid/quantized pools are unvalidated"
                )
        return result

    def init_cuda_graphs(self):
        super().init_cuda_graphs()
        vocab = int(self._target_worker.model_runner.model_config.vocab_size)
        self._dense_q_buffer(int(get_schedule().max_running_requests or 1), vocab)
        if self._dspine_sampler.optimized:
            from sglang_dspine.verification.sps import RaggedSpsCaps

            self.verify_caps = RaggedSpsCaps(self)
            from sglang_dspine.sampling.proposal import prepare

            self._dspine_sampler.proposal_cache = prepare(self._dspine_sampler)

    def _maybe_build_draft_sampler(self):
        return self._dspine_sampler

    def _greedy_sample_from_vocab_parallel_head(
        self, *, hidden_states: torch.Tensor, lm_head, chunk_size: int = 256
    ) -> torch.Tensor:
        """Seam B, the eager proposal path (a batch the draft graph does not take): the whole
        refinement tail, never an argmax.  hidden_states [bs*(T-1), H] are rows 1..T-1 of each
        block; the anchors come from the round's block ids."""
        sampler = self._dspine_sampler
        gamma = int(self.block_size) - 1
        n = int(hidden_states.shape[0])
        if n % gamma:
            raise RuntimeError("DSPINE eager proposal expects whole request blocks")
        bs = n // gamma
        if bs != sampler.round_bs:
            raise RuntimeError(
                f"DSPINE eager proposal for batch {bs} while the staged round has batch {sampler.round_bs}"
            )
        sampler.propose(hidden_states, self._draft_block_ids_buf[:bs].reshape(-1))
        if sampler.round_samples:
            sampler.sample(bs)
        if self.dspine_round_counts["eager"] == 0 and self._target_tp_rank == 0:
            logger.info("DSPINE: first eager draft round (bs=%d)", bs)
        self.dspine_round_counts["eager"] += 1
        return sampler.out[:n]

    def _propose_selector_block(
        self, *, draft_logits_output, bs: int, lm_head, anchor_token_ids, sampling_info
    ) -> torch.Tensor:
        """Eager path for batches the draft graph does not take: the whole refinement tail."""
        sampler = self._dspine_sampler
        hidden = draft_logits_output.hidden_states
        if hidden is None:
            raise RuntimeError("DSPINE draft model returned no hidden states")
        t = int(self.block_size)
        gamma = t - 1
        # The tail reads only each block's anchor (column 0) of the block ids.
        block_ids = torch.zeros((bs, t), dtype=torch.int64, device=hidden.device)
        block_ids[:, 0] = anchor_token_ids
        # The parent stages only when the draft graph was captured; stage in every case.
        sampler.stage_sampling_params(bs=bs, sampling_info=sampling_info)
        sampler(hidden, block_ids.view(-1))
        if _is_all_greedy(sampling_info):
            return sampler.out[: bs * gamma].view(bs, gamma)
        sampler.sample(bs)
        self._selector_sample = (sampler.candidate_out[:bs], sampler.q_out[:bs])
        return sampler.out[: bs * gamma].view(bs, gamma)

    def _selector_sampling_accept(
        self,
        *,
        candidates: torch.Tensor,
        next_token_logits: torch.Tensor,
        candidate_ids: torch.Tensor,
        q_rows: torch.Tensor,
        sampling_info,
        draft_input,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # q must be the distribution this round's drafts were drawn from; a round whose sampled
        # chain never ran (e.g. a draft graph replayed by another runner class) would verify
        # greedy drafts against a stale q.
        if not self._dspine_sampler.round_sampled:
            raise RuntimeError(
                "DSPINE sampled verify reached without this round's sampled draft chain"
            )
        return self._dspine_verify(
            candidates=candidates,
            next_token_logits=next_token_logits,
            candidate_ids=candidate_ids,
            q_rows=q_rows,
            sampling_info=sampling_info,
        )

    def _validate_phase1_sampling_support(self, batch) -> None:
        """Stage sampler state before each draft forward."""
        if self.target_distribution is not None:
            self.target_distribution.begin(batch)
        sampling_info = batch.sampling_info
        sampled = not _is_all_greedy(sampling_info)
        mode = batch.forward_mode
        if mode.is_extend() or batch.is_extend_in_batch or mode.is_idle():
            return
        self._dspine_sampler.random.stage(
            getattr(sampling_info, "sampling_seed", None) if sampled else None,
            batch.seq_lens,
        )
        self._dspine_sampler.stage_sampling_params(
            bs=len(batch.seq_lens), sampling_info=sampling_info
        )
        self.dspine_round_counts["staged"] += 1
        if sampled:
            if self.dspine_round_counts["sampled"] == 0 and self._target_tp_rank == 0:
                logger.info("DSPINE: first decode round with a sampled request")
            self.dspine_round_counts["sampled"] += 1

    def _dense_q_buffer(self, bs: int, vocab: int) -> torch.Tensor:
        """Persistent zeroed [cap, T-1, V] fp32 buffer for the dense q.  Sized once for the largest
        decode batch (max_running_requests) in init_cuda_graphs, and grown by doubling only if a
        batch ever exceeds it."""
        gamma = int(self.block_size) - 1
        buf = self._dspine_dense_q
        if buf is None or buf.shape[0] < bs or tuple(buf.shape[1:]) != (gamma, vocab):
            if buf is not None and (not self._warned_dense_q_realloc):
                logger.warning(
                    "DSPINE reallocates its dense-q buffer [%d, %d, %d] -> batch %d, vocab %d (memory outside the KV-pool budget)",
                    *buf.shape,
                    bs,
                    vocab,
                )
                self._warned_dense_q_realloc = True
            cap = max(bs, int(get_schedule().max_running_requests or 0))
            if buf is not None and tuple(buf.shape[1:]) == (gamma, vocab):
                cap = max(cap, 2 * buf.shape[0])
            buf = None
            self._dspine_dense_q = None
            buf = torch.zeros(
                (cap, gamma, vocab), dtype=torch.float32, device=self.device
            )
            self._dspine_dense_q = buf
        return buf

    def _accept_sampled_round(
        self,
        *,
        candidates: torch.Tensor,
        next_token_logits: torch.Tensor,
        sampling_info,
        max_top_k=None,
        uniform_top_k_value=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Seam C for this worker: verify a round with a sampled request against the q its drafts
        were drawn from.  A round whose sampled chain never ran would verify greedy drafts against
        a stale q, so it is refused."""
        sampler = self._dspine_sampler
        bs = int(candidates.shape[0])
        if not sampler.round_sampled:
            raise RuntimeError(
                "DSPINE sampled verify reached without this round's sampled draft chain"
            )
        if bs != sampler.round_bs:
            raise RuntimeError(
                f"DSPINE sampled verify for batch {bs} while the staged round has batch {sampler.round_bs}"
            )
        return self._dspine_verify(
            candidates=candidates,
            next_token_logits=next_token_logits,
            candidate_ids=sampler.candidate_out[:bs],
            q_rows=sampler.q_out[:bs],
            sampling_info=sampling_info,
            max_top_k=max_top_k,
            uniform_top_k_value=uniform_top_k_value,
        )

    def _dspine_verify(
        self,
        *,
        candidates,
        next_token_logits,
        candidate_ids,
        q_rows,
        sampling_info,
        max_top_k=None,
        uniform_top_k_value=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = candidates.shape[0]
        target_probs = None
        if self.target_distribution is not None:
            target_probs = self.target_distribution.dense(
                next_token_logits, sampling_info, bs
            )
        return dspine_accept_sampling(
            target_probs=target_probs,
            candidates=candidates,
            target_logits=next_token_logits,
            candidate_ids=candidate_ids,
            q_rows=q_rows,
            sampling_info=sampling_info,
            sampling_backend=get_exec().kernel.sampling_backend,
            dense_q=self._dense_q_buffer(bs, int(next_token_logits.shape[-1])),
            random_state=getattr(self._dspine_sampler, "random", None),
        )


def accept_dspine_sampled_round(
    *,
    candidates,
    next_token_logits,
    sampling_info,
    max_top_k=None,
    uniform_top_k_value=None,
    **_dflash_only_arguments,
):
    """Seam C under DSPINE (called by the plugin's AROUND hook in place of DFlash's
    compute_dflash_sampling_correct_drafts_and_bonus): (accept_len int32 [bs], bonus int64 [bs]).
    Sampling follows the request's ordinary AR target distribution."""
    worker = _active_worker
    if worker is None:
        raise RuntimeError(
            "DSPINE sampled verify reached before the DSpine worker exists"
        )
    return worker._accept_sampled_round(
        candidates=candidates,
        next_token_logits=next_token_logits,
        sampling_info=sampling_info,
        max_top_k=max_top_k,
        uniform_top_k_value=uniform_top_k_value,
    )
