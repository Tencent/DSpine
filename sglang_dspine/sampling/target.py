"""Native target distributions with sparse-q chain rejection for any batch."""

import copy

import torch

from sglang_dspine.sampling.sparse import accept


class TargetDistribution:
    def __init__(self, worker):
        self.worker = worker
        self.scratch = None

    def dense(self, logits, info, bs, row_to_request=None):
        from sglang.srt.runtime_context import get_exec

        from sglang_dspine.runtime.worker import build_dspine_target_probs

        width = int(self.worker.block_size)
        if row_to_request is not None:
            info = copy.copy(info)
            for name in ("temperatures", "top_ks", "top_ps", "min_ps"):
                setattr(info, name, getattr(info, name).index_select(0, row_to_request))
            bs, width = logits.shape[0], 1
        return build_dspine_target_probs(
            next_token_logits=logits,
            sampling_info=info,
            block_size=width,
            bs=bs,
            sampling_backend=get_exec().kernel.sampling_backend,
        )

    def sample_prefix(
        self,
        *,
        candidates,
        next_token_logits,
        sampling_info,
        prefix_lens,
        bs,
        verify_lens,
        row_offsets=None,
        row_to_request=None,
    ):
        sampler = self.worker._dspine_sampler
        if not sampler.round_sampled or sampler.round_bs != bs:
            raise RuntimeError(
                "Sparse rejection must consume this round's actual proposal q"
            )
        # Other block layouts retain the worker's generic rejection path.
        width = int(self.worker.block_size)
        if width != 16 or sampler.k != 16:
            return None
        if candidates.shape != (bs, width):
            raise ValueError("Proposal storage does not match this decode batch")
        sparse = None
        if next_token_logits.device.type == "cuda":
            from sglang.srt.runtime_context import get_exec
            from sglang_dspine.sampling.sparse.target import build_sparse_target

            sparse = build_sparse_target(
                next_token_logits, sampling_info, row_to_request=row_to_request,
                row_offsets=row_offsets, verify_lens=verify_lens, block_size=width,
                sampling_backend=get_exec().kernel.sampling_backend,
            )
        if sparse is None:
            p = (
                self.dense(next_token_logits, sampling_info, bs)
                if row_to_request is None
                else self.dense(next_token_logits, sampling_info, bs, row_to_request)
            )
            flat_p = p.reshape(next_token_logits.shape[0], -1)
        else:
            flat_p = sparse.dense_p
        # Unfiltered p is already dense: retain sparse q without scanning and
        # compressing a support that will overflow. Filtered supports may pack.
        filtered = any(
            getattr(sampling_info, name)
            for name in (
                "need_top_k_sampling", "need_top_p_sampling", "need_min_p_sampling"
            )
        )
        packed = (
            sparse.packed if sparse is not None
            else accept.compress_probs(torch, flat_p, capacity=32) if filtered else None
        )
        outputs = self.worker._next_accept_bonus_buffers(bs)
        from sglang_dspine.sampling.random import verification_uniforms

        uniforms, final = verification_uniforms(
            bs, width, candidates.device, sampler.random
        )
        if (
            self.scratch is None
            or self.scratch.device != candidates.device
            or self.scratch.shape[0] < bs
        ):
            capacity = max(
                bs, 2 * self.scratch.shape[0] if self.scratch is not None else 1
            )
            self.scratch = torch.empty(
                (capacity, 4096), device=candidates.device, dtype=torch.float32
            )
        accept.accept_sparse(
            torch,
            candidates=candidates,
            dense_p=flat_p,
            packed=packed,
            q_ids=sampler.candidate_out[:bs],
            q_values=sampler.q_out[:bs],
            verify_lens=verify_lens,
            uniforms=uniforms,
            uniform_final=final,
            prefix_lens=prefix_lens,
            outputs=outputs,
            scratch=self.scratch[:bs],
            row_offsets=row_offsets,
            vocab_size=next_token_logits.shape[-1],
            denominator=sparse.denominator if sparse is not None else None,
        )
        return outputs
