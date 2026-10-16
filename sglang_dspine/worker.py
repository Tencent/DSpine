"""DSpine speculative decoding worker for SGLang's DFlash execution path.

The draft CUDA graph captures candidate selection, the greedy chain, and refinement-state
copies. Sampled rounds run their fp32 serial chain eagerly after replay or an eager draft
forward, before the parent consumes the proposal.

Requires TP=1, a checkpoint-fixed block size, no compact draft window, an unquantized
BF16 target LM head, and TF32 disabled.
"""

import logging
from typing import Optional, Tuple

import torch
from sglang.kernels.ops.speculative.dspark.dspark_accept import SoftmaxTemp
from sglang.kernels.ops.speculative.reject_sampling import (
    chain_speculative_sampling_triton,
)
from sglang.srt.layers.sampler import top_k_top_p_min_p_sampling_from_probs_torch
from sglang.srt.lora.layers import unwrap_lora_layer
from sglang.srt.runtime_context import get_exec, get_parallel, get_schedule
from sglang.srt.speculative.dflash_utils import (
    _dflash_top_k_renorm_prob,
    _dflash_top_p_renorm_prob,
    _get_or_create_chain_verify_buffers,
)
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2, _is_all_greedy

from sglang_dspine import kernels as dspine_kernels
from sglang_dspine.models.dspine import DSpineDraftModel

logger = logging.getLogger(__name__)


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
        if torch.get_float32_matmul_precision() != "highest" or bool(
            torch.backends.cuda.matmul.allow_tf32
        ):
            raise RuntimeError(
                "DSPINE computes its sampling table in true fp32: TF32 matmuls must be "
                "off (do not enable TF32 for this server)"
            )
        bridge = draft_model.selected_prefix_bridge
        if not bridge._ready:
            raise RuntimeError(
                "DSPINE sampler built before the draft weights were prepared"
            )
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
        # Vocabulary-wide greedy tables: no GEMM at decode time beyond the pass-1 head.
        self.tg, self.tsv = dspine_kernels.build_serial_tables(
            lm_head_weight,
            draft_model.norm.weight.detach(),
            bridge.feature_write_weights[last].detach(),
            bridge.packed_weight[r:].detach(),
            bridge.codebook,
        )
        # fp32 copies for the sampling table (converted once, not per step).
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
        # Staged per request before every round; defaults are a valid greedy setting.
        self.inv_temp = torch.ones((self.max_bs,), dtype=f32, device=device)
        self.top_k = torch.full((self.max_bs,), k, dtype=torch.int32, device=device)
        self.top_p = torch.ones((self.max_bs,), dtype=f32, device=device)
        self.min_p = torch.zeros((self.max_bs,), dtype=f32, device=device)
        self.greedy = torch.ones((self.max_bs,), dtype=torch.int32, device=device)
        # Refinement state copied out of the draft forward (graph-pool memory is not ours).
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
        # The open round (host side): its batch, whether it has a sampled request, and whether
        # its sampled chain has run.
        self._round_bs = 0
        self._round_samples = False
        self._round_sampled = False

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
        if not self._round_samples:
            self.greedy[:bs].fill_(1)
            return
        top_ks = sampling_info.top_ks.view(-1)[:bs]
        torch.reciprocal(
            torch.clamp(
                sampling_info.temperatures.view(-1)[:bs].to(torch.float32), min=1e-5
            ),
            out=self.inv_temp[:bs],
        )
        # top_k >= K keeps every candidate; SGLang encodes "no top-k" as a huge top_k.
        self.top_k[:bs].copy_(torch.clamp(top_ks, min=1, max=self.k).to(torch.int32))
        self.top_p[:bs].copy_(sampling_info.top_ps.view(-1)[:bs].to(torch.float32))
        self.min_p[:bs].copy_(sampling_info.min_ps.view(-1)[:bs].to(torch.float32))
        self.greedy[:bs].copy_((top_ks <= 1).to(torch.int32))

    @torch.no_grad()
    def __call__(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> None:
        """Capture-safe part: candidates, the greedy chain, and the state copy."""
        dm = self.draft_model
        stash = dm.last_write_stash
        if stash is None:
            raise RuntimeError(
                "DSPINE tail ran without the draft forward's last-write stash"
            )
        bridge = dm.selected_prefix_bridge
        t, k, r = self.block_size, self.k, int(bridge.rank)
        m, h = hidden_states.shape
        if m % t:
            raise RuntimeError("DSPINE tail expects whole request blocks")
        bs = m // t
        n = bs * (t - 1)
        if bs > self.max_bs:
            raise RuntimeError(
                f"DSPINE batch {bs} exceeds the sampler capacity {self.max_bs}"
            )
        rows = hidden_states.view(bs, t, h)[:, 1:, :].reshape(n, h)
        logits = torch.matmul(rows, self.lm_head_weight.T)
        cand = dspine_kernels.topk_ids(logits, k)
        del logits
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
        )
        dspine_kernels.serial_chain(idx, cand, self.greedy_out, t)
        # out is what the round proposes: the greedy chain unless the sampled step replaces it.
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
        t, k = self.block_size, self.k
        m, n = bs * t, bs * (t - 1)
        cand = self.candidate_out[:bs].view(n, k)
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
        )
        self.uniforms[:n].uniform_()
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
                f"DSPINE draft replay for batch {int(forward_batch.batch_size)} while the "
                f"staged round has batch {self._round_bs}"
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
      pytorch: the filtered weights of top_k_top_p_min_p_sampling_from_probs_torch itself.
    Rejection sampling against this p makes the committed tokens follow non-speculative sampling."""
    t = int(block_size)
    probs = SoftmaxTemp.execute(
        logits=next_token_logits,
        temperatures=sampling_info.temperatures,
        rows_per_request=t,
    ).to(torch.float32)
    need_k = bool(sampling_info.need_top_k_sampling)
    need_p = bool(sampling_info.need_top_p_sampling)
    need_min_p = bool(sampling_info.need_min_p_sampling)
    if not (need_k or need_p or need_min_p):
        return probs.view(bs, t, -1)
    top_ks = torch.repeat_interleave(sampling_info.top_ks.view(-1), t)
    top_ps = torch.repeat_interleave(sampling_info.top_ps.view(-1), t)
    min_ps = torch.repeat_interleave(sampling_info.min_ps.view(-1), t)
    if sampling_backend == "pytorch":
        _, weights, token_ids, _ = top_k_top_p_min_p_sampling_from_probs_torch(
            probs,
            top_ks,
            top_ps,
            min_ps,
            need_min_p,
            None,
            None,
            return_filtered_probs=True,
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
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs.view(bs, t, -1)


def dspine_accept_sampling(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    candidate_ids: torch.Tensor,
    q_rows: torch.Tensor,
    sampling_info,
    sampling_backend: str,
    dense_q: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Verify a drafted block with chain rejection sampling.

    Accept a draft token x with probability min(1, p(x) / q(x)). On the first rejection,
    sample its replacement from normalise(max(0, p - q)); if all drafts are accepted, draw
    the bonus from p. ``build_dspine_target_probs`` supplies the target distribution p used
    by SGLang's ordinary sampler for the configured backend.

    q_rows must contain the distribution used to sample each token in the current round,
    including one-hot rows on the greedy chain tokens for greedy requests in mixed batches.

    candidates [bs, T] = [anchor, T-1 drafts]; candidate_ids / q_rows [bs, T-1, K] give the sparse
    q each draft was sampled from (draft i = candidates[:, i+1] belongs to row i); dense_q is a
    zeroed [>=bs, T-1, V] fp32 buffer, scattered into and zeroed back here.  Returns
    (accept_len int32 [bs], bonus int64 [bs])."""
    bs, block = candidates.shape
    gamma = block - 1
    device = candidates.device
    if (
        tuple(candidate_ids.shape[:2]) != (bs, gamma)
        or candidate_ids.shape != q_rows.shape
    ):
        raise RuntimeError("DSPINE verify: candidate / q layout mismatch")
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
    uniform_samples = torch.rand((bs, gamma), dtype=torch.float32, device=device)
    uniform_final = torch.rand((bs,), dtype=torch.float32, device=device)
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
        # candidate_ids may view a buffer the next draft step overwrites: zero back now.
        draft_probs.scatter_(-1, candidate_ids, 0.0)
    accept_len = accept_token_num.clone()
    rows = torch.arange(bs, dtype=torch.int64, device=device)
    accept_pos = accept_index[rows, accept_len.to(torch.int64)].to(torch.int64)
    bonus = predicts[accept_pos].to(torch.int64)
    return accept_len.to(torch.int32), bonus


class DSpineWorkerV2(DFlashWorkerV2):
    """DFlashWorkerV2 with DSpine draft sampling and sampled verification."""

    def __init__(self, server_args, gpu_id: int, ps, nccl_port: int, target_worker):
        # SGLang 0.5.20 builds draft workers with the scheduler's ParallelState (ps), as for
        # DFlashWorkerV2; ranks are read from it (ModelRunner has no tp_rank in 0.5.20).
        super().__init__(server_args, gpu_id, ps, nccl_port, target_worker)
        dm = self.draft_model
        if not isinstance(dm, DSpineDraftModel):
            raise ValueError(
                "--speculative-algorithm DSPINE loaded draft class "
                f"{type(dm).__name__}; expected DSpineDraftModel"
            )
        if int(get_parallel().tp_group.world_size) != 1:
            raise ValueError("DSPINE supports tp=1 only")
        if self.use_compact_draft_cache:
            raise ValueError("DSPINE does not support --speculative-draft-window-size")
        if self._is_domino or self.selector is not None:
            raise ValueError("DSPINE drafts carry no Domino / selector head")
        if int(self.block_size) != int(dm.block_size):
            raise ValueError(
                "DSPINE needs --speculative-num-draft-tokens == draft block size"
            )
        target_model = self._target_worker.model_runner.model
        lm_head = unwrap_lora_layer(getattr(target_model, "lm_head", None))
        weight = getattr(lm_head, "weight", None)
        if (
            weight is None
            or weight.dtype != torch.bfloat16
            or weight.dim() != 2
            or weight.stride(1) != 1
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
                f"DSPINE: target vocabulary {num_org} != code-bridge codebook "
                f"{int(bridge.codebook.shape[0])}"
            )
        if int(weight.shape[1]) != int(dm.config.hidden_size):
            raise ValueError("DSPINE: target lm_head width != draft hidden size")
        graph_bs = list(get_exec().graph.cuda_graph_config.decode.bs or [])
        # ModelRunner.max_running_requests is only set later (KV-pool sizing); the configured
        # schedule value is its upper bound.
        max_bs = max(graph_bs + [int(get_schedule().max_running_requests or 0), 1])
        self._dspine_sampler = DSpineDraftSampler(
            draft_model=dm,
            lm_head_weight=weight[:num_org],
            block_size=self.block_size,
            max_bs=max_bs,
            device=self.device,
        )
        self._dspine_dense_q: Optional[torch.Tensor] = None
        self._warned_sampling_seed = False
        # Ride DFlashWorkerV2's candidate-selector path (read only inside dflash_worker_v2.py):
        # with `selector` set, the parent stages sampling params on the draft sampler, proposes
        # `_draft_sampler.out` after a graph replay or calls _propose_selector_block when the
        # batch runs eagerly, and routes a round with a sampled request to
        # _selector_sampling_accept.  The draft graph runner reaches the sampler through the
        # draft model to run the sampled chain right after each replay.
        self.selector = self._dspine_sampler
        self._selector_sampling_enabled = True
        dm.dspine_sampler = self._dspine_sampler
        if self.ps.tp_rank == 0:
            logger.info(
                "DSPINE ready: block_size=%d head_topk=%d bridge=%s sampler_capacity_bs=%d "
                "verify: greedy=argmax chain, sampled=chain rejection sampling over the sampled "
                "serial chain (temperature/top_k/top_p/min_p)",
                self.block_size,
                self._dspine_sampler.k,
                bridge.implementation,
                max_bs,
            )

    # -- draft side ---------------------------------------------------------------------------

    def _maybe_build_draft_sampler(self):
        return self._dspine_sampler

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

    # -- verify side --------------------------------------------------------------------------

    def _validate_phase1_sampling_support(self, batch) -> None:
        sampling_info = batch.sampling_info
        if (
            sampling_info is not None
            and not sampling_info.is_all_greedy
            and sampling_info.sampling_seed is not None
            and not self._warned_sampling_seed
            and self.ps.tp_rank == 0
        ):
            logger.warning(
                "DSPINE samples with the device RNG; per-request sampling_seed is not "
                "reproduced (the output distribution is unaffected)."
            )
            self._warned_sampling_seed = True

    def _dense_q_buffer(self, bs: int, vocab: int) -> torch.Tensor:
        """Persistent zeroed [cap, T-1, V] fp32 buffer for the dense q.  Sized once for the largest
        decode batch (max_running_requests) -- the pre-warm allocates it before post-capture KV
        sizing -- and grown by doubling only if a batch ever exceeds it."""
        gamma = int(self.block_size) - 1
        buf = self._dspine_dense_q
        if buf is None or buf.shape[0] < bs or tuple(buf.shape[1:]) != (gamma, vocab):
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

    def _dspine_verify(
        self, *, candidates, next_token_logits, candidate_ids, q_rows, sampling_info
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = candidates.shape[0]
        return dspine_accept_sampling(
            candidates=candidates,
            target_logits=next_token_logits,
            candidate_ids=candidate_ids,
            q_rows=q_rows,
            sampling_info=sampling_info,
            sampling_backend=get_exec().kernel.sampling_backend,
            dense_q=self._dense_q_buffer(bs, int(next_token_logits.shape[-1])),
        )

    def _measure_sampling_peak(
        self, *, candidates, target_logits, sampling_info
    ) -> int:
        """Rehearse DSpine's sampled verify at the pre-warm batch in its heaviest filter paths
        (joint top-k/top-p, and the min-p branch) and return the larger peak; the first call also
        allocates the persistent dense-q buffer before post-capture KV sizing."""
        import dataclasses

        bs, block = candidates.shape
        gamma, k = block - 1, self._dspine_sampler.k
        device = candidates.device
        candidate_ids = (
            torch.arange(gamma * k, dtype=torch.int64, device=device)
            .view(1, gamma, k)
            .expand(bs, gamma, k)
            .contiguous()
        )
        q_rows = torch.full((bs, gamma, k), 1.0 / k, dtype=torch.float32, device=device)
        top_ks = torch.full((bs,), 20, dtype=torch.int32, device=device)
        variants = (
            dataclasses.replace(
                sampling_info,
                top_ks=top_ks,
                need_top_k_sampling=True,
                need_top_p_sampling=True,
            ),
            dataclasses.replace(
                sampling_info,
                top_ks=top_ks,
                min_ps=torch.full((bs,), 0.05, dtype=torch.float32, device=device),
                need_top_k_sampling=True,
                need_top_p_sampling=True,
                need_min_p_sampling=True,
            ),
        )
        self._dense_q_buffer(bs, int(target_logits.shape[-1]))
        peak = 0
        for info in variants:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base_bytes = torch.cuda.memory_stats()["allocated_bytes.all.current"]
            self._dspine_verify(
                candidates=candidates,
                next_token_logits=target_logits,
                candidate_ids=candidate_ids,
                q_rows=q_rows,
                sampling_info=info,
            )
            torch.cuda.synchronize()
            peak = max(
                peak, torch.cuda.memory_stats()["allocated_bytes.all.peak"] - base_bytes
            )
        return peak
