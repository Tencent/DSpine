"""Exact target-distribution construction and sparse rejection for T=1/K20/P.95."""

import torch

from sglang_dspine.sampling.sparse import accept, materializer, producer
from sglang_dspine.sampling.sparse.graphs import GraphP4Cache


def _signature(tensor):
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
        tensor._version,
    )


def _params(info):
    return tuple(
        _signature(getattr(info, key))
        for key in ("temperatures", "top_ks", "top_ps", "min_ps")
    )


class TargetDistribution:
    def __init__(self, worker):
        self.worker = worker
        self.graphs = None
        self.certificate = None
        self.scratch = {}

    def prepare(self, stream):
        from sglang_dspine.runtime.worker import build_dspine_target_probs

        props = torch.cuda.get_device_properties(self.worker.device)
        if (
            props.major,
            props.minor,
            props.multi_processor_count,
            props.max_threads_per_multi_processor,
        ) != (9, 0, 132, 2048):
            return
        producer.initialize()
        self.graphs = GraphP4Cache(
            torch, producer, materializer, build_dspine_target_probs
        )
        with torch.cuda.stream(stream):
            for bs in range(1, 8):
                logits = torch.zeros(
                    (bs * 16, 248320), device=self.worker.device, dtype=torch.float32
                )
                self.graphs.prepare(logits, "dense" if bs <= 3 else "sparse")

    def begin(self, batch):
        self.certificate = None
        if (
            batch is None
            or not batch.forward_mode.is_decode()
            or batch.is_extend_in_batch
        ):
            return
        info = batch.sampling_info
        if (
            batch.has_grammar
            or batch.return_logprob
            or not 1 <= len(batch.reqs) <= 8
            or info.is_all_greedy
            or info.is_any_greedy
            or info.need_min_p_sampling
            or not info.need_top_k_sampling
            or not info.need_top_p_sampling
            or info.has_custom_logit_processor
            or info.logit_bias is not None
            or any(x is not None for x in (info.grammars or ()))
        ):
            return
        if any(
            (
                r.sampling_params.temperature,
                r.sampling_params.top_k,
                r.sampling_params.top_p,
                r.sampling_params.min_p,
            )
            != (1.0, 20, 0.95, 0.0)
            for r in batch.reqs
        ):
            return
        self.certificate = (info, len(batch.reqs), _params(info))

    def eligible(self, logits, info, bs):
        from sglang.srt.runtime_context import get_exec

        c = self.certificate
        return (
            self.graphs is not None
            and c is not None
            and c[0] is info
            and c[1] == bs
            and c[2] == _params(info)
            and get_exec().kernel.sampling_backend == "flashinfer"
            and logits.is_cuda
            and logits.is_contiguous()
            and logits.dtype == torch.float32
            and logits.shape == (bs * 16, 248320)
        )

    def dense(self, logits, info, bs):
        if bs <= 3 and self.eligible(logits, info, bs):
            return self.graphs.run(logits, "dense").view(bs, 16, -1)
        return None

    def sample_prefix(
        self,
        *,
        candidates,
        next_token_logits,
        sampling_info,
        prefix_lens,
        bs,
        verify_lens,
    ):
        if not 4 <= bs <= 8 or not self.eligible(next_token_logits, sampling_info, bs):
            return None
        from sglang_dspine.runtime.worker import build_dspine_target_probs

        sampler = self.worker._dspine_sampler
        if not sampler.round_sampled or sampler.round_bs != bs:
            raise RuntimeError(
                "Sparse rejection must consume this round's actual proposal q"
            )
        if bs < 8:
            bundle = self.graphs.run(next_token_logits, "sparse")
        else:
            bundle = producer.build_sparse_joint(
                logits=next_token_logits,
                info=sampling_info,
                bs=bs,
                width=16,
                native=build_dspine_target_probs,
            )
        outputs = self.worker._next_accept_bonus_buffers(bs)
        # Preserve the native draw order and sizes; q is never recomputed here.
        from sglang_dspine.sampling.random import verification_uniforms

        uniforms, final = verification_uniforms(
            bs, 16, candidates.device, sampler.random
        )
        key = (bs, candidates.device)
        if key not in self.scratch:
            self.scratch[key] = torch.empty(
                (bs, 4096), device=candidates.device, dtype=torch.float32
            )
        packed = (bundle["ids"], bundle["probs"], bundle["lengths"], bundle["overflow"])
        accept.accept_sparse(
            torch,
            candidates=candidates,
            dense_p=bundle["raw_masked"],
            packed=packed,
            q_ids=sampler.candidate_out[:bs],
            q_values=sampler.q_out[:bs],
            verify_lens=verify_lens,
            uniforms=uniforms,
            uniform_final=final,
            prefix_lens=prefix_lens,
            outputs=outputs,
            scratch=self.scratch[key],
            denominator=bundle["denominator"],
            raw_fallback=True,
        )
        return outputs


def after_scheduler_init(result, scheduler, *args, **kwargs):
    worker = getattr(scheduler, "draft_worker", None)
    distribution = getattr(worker, "target_distribution", None)
    if distribution is not None:
        distribution.prepare(scheduler.forward_stream)
    return result
