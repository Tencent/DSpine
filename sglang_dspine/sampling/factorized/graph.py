"""Cached proposal graphs for small batches and bounded-workspace eager batches."""

import logging

import torch

from sglang_dspine.sampling.factorized.kernels import Workspace, factor_table
from sglang_dspine.sampling.factorized.weights import build_cache

logger = logging.getLogger(__name__)


def _signature(tensor):
    if tensor is None:
        return None
    try:
        version = tensor._version
    except RuntimeError:
        version = None
    return (
        tensor.data_ptr(),
        tensor.device,
        tensor.dtype,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        version,
    )


class ProposalGraphs:
    def __init__(self, sampler):
        self.sampler = sampler
        self.entries = {}
        self.workspaces = {}
        self.base = {}
        self.stream = None
        self.eager_batches = set()
        self.graph_max_bs = min(8, sampler.max_bs)
        self.eager_output = (
            torch.empty(
                (sampler.max_bs * 15, 16, 16),
                device=sampler.raw.device,
                dtype=torch.float32,
            )
            if sampler.max_bs > self.graph_max_bs
            else None
        )
        bridge = sampler.draft_model.selected_prefix_bridge
        tensors = (
            bridge.codebook,
            sampler.pred_gate_f32,
            sampler.prev_f32,
            sampler.gate_b_f32,
            sampler.write_f32,
            sampler.norm_f32,
            sampler.lm_head_weight,
        )
        self.source_tensors = tensors
        self.source_signature = tuple(_signature(x) for x in tensors)
        self.weights = build_cache(*tensors)
        self.weight_signature = tuple(
            (key, _signature(value)) for key, value in sorted(self.weights.items())
        )
        current = torch.cuda.current_stream(sampler.raw.device)
        for bs in range(1, self.graph_max_bs + 1):
            n = bs * 15
            self.base[bs] = torch.empty(
                (n, 16), device=sampler.raw.device, dtype=torch.float64
            )
            self.workspaces[bs] = Workspace.allocate(
                n,
                16,
                sampler.raw.shape[1],
                sampler.query.shape[1],
                sampler.raw.device,
            )
            output = torch.empty(
                (n, 16, 16), device=sampler.raw.device, dtype=torch.float32
            )
            stream = torch.cuda.Stream(device=sampler.raw.device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._compute(bs, output)
            current.wait_stream(stream)
            current.synchronize()
            graph = torch.cuda.CUDAGraph()
            pool = torch.cuda.graph_pool_handle()
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                self._compute(bs, output)
            current.wait_stream(stream)
            current.synchronize()
            self.entries[bs] = graph, output, pool

    def _compute(self, bs, output, start=0):
        sampler = self.sampler
        end = start + bs
        return factor_table(
            sampler.raw[start * 16 : end * 16],
            sampler.rms[start * 16 : end * 16],
            sampler.query[start * 16 : end * 16],
            sampler.anchors[start:end],
            sampler.candidate_out[start:end].view(bs * 15, 16),
            self.weights,
            self.sampler.norm_eps,
            16,
            output,
            self.base[bs],
            self.workspaces[bs],
        )

    def lookup(self, bs):
        if not 1 <= bs <= self.sampler.max_bs:
            raise ValueError(
                f"Proposal batch {bs} exceeds capacity {self.sampler.max_bs}"
            )
        if tuple(_signature(x) for x in self.source_tensors) != self.source_signature:
            raise RuntimeError("Proposal weights changed after graph capture")
        signature = tuple((k, _signature(v)) for k, v in sorted(self.weights.items()))
        if signature != self.weight_signature:
            raise RuntimeError("Factorized proposal cache changed after graph capture")
        current = torch.cuda.current_stream(self.sampler.raw.device)
        stream = int(current.cuda_stream)
        if self.stream is None:
            self.stream = stream
        elif stream != self.stream:
            raise RuntimeError(
                "Proposal graphs must stay on their original inference stream"
            )
        if bs in self.entries:
            graph, output, _ = self.entries[bs]
            graph.replay()
            return output
        # Each request's table is independent. Reuse the existing small-batch
        # workspaces on the inference stream rather than retaining a graph and
        # a full FP64 cancellation workspace for every larger batch size.
        output = self.eager_output[: bs * 15]
        for start in range(0, bs, self.graph_max_bs):
            count = min(self.graph_max_bs, bs - start)
            self._compute(count, output[start * 15 : (start + count) * 15], start)
        if bs not in self.eager_batches:
            logger.info(
                "DSpine eager proposal table: batch=%d chunk=%d", bs, self.graph_max_bs
            )
            self.eager_batches.add(bs)
        return output
