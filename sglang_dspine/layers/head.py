"""FP8 retrieval followed by rescoring with the original BF16 vocabulary head."""

import threading

import torch

from sglang_dspine.layers.fp8 import gather, padding
from sglang_dspine.layers.fp8.coarse import FullVFP8Coarse
from sglang_dspine.layers.fp8.workspace import PrivateFIWorkspace


class DraftHead:
    def __init__(self, sampler):
        self.sampler = sampler
        self.coarse = FullVFP8Coarse(torch, sampler.lm_head_weight)
        self.actual_bs = torch.full(
            (1,), sampler.max_bs, device=sampler.raw.device, dtype=torch.int32
        )
        self.rows = torch.empty(
            (sampler.max_bs * 15, 5120),
            device=sampler.raw.device,
            dtype=torch.bfloat16,
        )
        self.capture_owner = None
        self.graph_owners = {}
        self.eager_owners = {}

    def retrieve(self, rows):
        from flashinfer.topk import top_k

        rows = padding.mask(rows, self.actual_bs, self.rows[: rows.shape[0]])
        if self.capture_owner is not None:
            return self.coarse.retrieve_and_rescore(
                rows, fi_top_k=top_k, gather_ops=gather
            )[0]
        key = (
            threading.get_ident(),
            int(torch.cuda.current_stream(rows.device).cuda_stream),
        )
        owner = self.eager_owners.get(key)
        if owner is None:
            owner = PrivateFIWorkspace(torch, torch.cuda.current_stream(rows.device))
            self.eager_owners[key] = owner
        with owner.bind():
            return self.coarse.retrieve_and_rescore(
                rows, fi_top_k=top_k, gather_ops=gather
            )[0]


def around_capture(original, backend, shape_key, forward_fn, *args, **kwargs):
    runner = backend._cuda_graph_runner
    from sglang_dspine.runtime.graph_runner import DSpineDraftCudaGraphRunner

    if not isinstance(runner, DSpineDraftCudaGraphRunner):
        return original(backend, shape_key, forward_fn, *args, **kwargs)
    sampler = runner.model_runner.model.dspine_sampler
    if not sampler.optimized:
        return original(backend, shape_key, forward_fn, *args, **kwargs)
    if sampler.head is None:
        sampler.head = DraftHead(sampler)
    head = sampler.head
    stream = backend._capture_stream
    if stream is None or head.capture_owner is not None:
        raise RuntimeError(
            "DSpine draft capture must own its CUDA stream and scratch buffers"
        )
    owner = PrivateFIWorkspace(torch, stream)
    head.capture_owner = owner
    try:
        with owner.bind():
            rows = torch.zeros(
                (int(shape_key.size) * 15, 5120),
                device=sampler.raw.device,
                dtype=torch.bfloat16,
            )
            for _ in range(3):
                warm = head.retrieve(rows)
            stream.synchronize()
            del rows, warm
            owner.freeze()
            result = original(backend, shape_key, forward_fn, *args, **kwargs)
        graph = backend._graphs[shape_key]
        graph._dspine_head_workspace = owner
        head.graph_owners[(id(backend), shape_key)] = owner
        return result
    finally:
        head.capture_owner = None


def around_cleanup(original, backend, *args, **kwargs):
    runner = backend._cuda_graph_runner
    from sglang_dspine.runtime.graph_runner import DSpineDraftCudaGraphRunner

    if not isinstance(runner, DSpineDraftCudaGraphRunner):
        return original(backend, *args, **kwargs)
    head = getattr(runner.model_runner.model.dspine_sampler, "head", None)
    if head is None:
        return original(backend, *args, **kwargs)
    torch.cuda.synchronize()
    result = original(backend, *args, **kwargs)
    for key in list(head.graph_owners):
        if key[0] == id(backend):
            head.graph_owners.pop(key).release_after_reset()
    return result
