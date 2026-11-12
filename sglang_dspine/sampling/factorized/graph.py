"""Factorized proposal logits with bounded graph and workspace storage."""

from dataclasses import fields, replace

import torch

from sglang_dspine.sampling.factorized.kernels import Workspace, factor_table
from sglang_dspine.sampling.factorized.weights import build_cache


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


def _capture_sizes(configured, capacity):
    """Select configured shapes with at most linear retained graph storage."""
    candidates = sorted({int(x) for x in configured if 0 < int(x) <= capacity})
    sizes = []
    for bs in candidates:
        if not sizes or bs >= 2 * sizes[-1]:
            sizes.append(bs)
    if candidates and candidates[-1] != sizes[-1]:
        sizes[-1] = candidates[-1]
    return sizes


class _Buffers:
    def __init__(self, bs, sampler):
        self.bs = bs
        n = bs * (sampler.block_size - 1)
        k = sampler.k
        h, r = sampler.raw.shape[1], sampler.query.shape[1]
        device = sampler.raw.device
        self.base = torch.empty((n, k), device=device, dtype=torch.float64)
        self.output = torch.empty((n, k, k), device=device, dtype=torch.float32)
        self.workspace = Workspace.allocate(n, k, h, r, device)

    def view(self, n):
        parts = {
            field.name: getattr(self.workspace, field.name)[:n]
            for field in fields(self.workspace)
            if field.name != "fallback"
        }
        parts["fallback"] = replace(
            self.workspace.fallback, y=self.workspace.fallback.y[:n]
        )
        return self.output[:n], self.base[:n], replace(self.workspace, **parts)


class ProposalGraphs:
    def __init__(self, sampler):
        self.sampler = sampler
        self.entries = {}
        self.stream = None
        self.graph_buffers = self.eager_buffers = None
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
        sizes = _capture_sizes(
            getattr(sampler, "capture_batch_sizes", ()), sampler.max_bs
        )
        if not sizes:
            return
        self.graph_buffers = _Buffers(max(sizes), sampler)
        current = torch.cuda.current_stream(sampler.raw.device)
        for bs in sizes:
            stream = torch.cuda.Stream(device=sampler.raw.device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._compute(bs, self.graph_buffers)
            current.wait_stream(stream)
            current.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                self._compute(bs, self.graph_buffers)
            current.wait_stream(stream)
            current.synchronize()
            self.entries[bs] = graph

    def _compute(self, bs, buffers):
        sampler = self.sampler
        t, k = sampler.block_size, sampler.k
        n = bs * (t - 1)
        output, base, workspace = buffers.view(n)
        return factor_table(
            sampler.raw[: bs * t],
            sampler.rms[: bs * t],
            sampler.query[: bs * t],
            sampler.anchors[:bs],
            sampler.candidate_out[:bs].view(n, k),
            self.weights,
            sampler.norm_eps,
            t,
            output,
            base,
            workspace,
        )

    def lookup(self, bs):
        if not 0 < bs <= self.sampler.max_bs:
            raise ValueError("Proposal batch exceeds the configured sampler capacity")
        if tuple(_signature(x) for x in self.source_tensors) != self.source_signature:
            raise RuntimeError("Proposal weights changed after cache preparation")
        signature = tuple((k, _signature(v)) for k, v in sorted(self.weights.items()))
        if signature != self.weight_signature:
            raise RuntimeError("Factorized proposal cache changed after preparation")
        current = torch.cuda.current_stream(self.sampler.raw.device)
        stream = int(current.cuda_stream)
        if self.stream is None:
            self.stream = stream
        elif stream != self.stream:
            raise RuntimeError("Proposal buffers must stay on their inference stream")
        bucket = next((size for size in sorted(self.entries) if size >= bs), None)
        if bucket is not None:
            if bucket != bs:
                # Requests are independent. Finite, valid padding lets a
                # captured bucket cover every smaller intermediate batch.
                sampler = self.sampler
                rows = slice(bs * sampler.block_size, bucket * sampler.block_size)
                sampler.raw[rows].zero_()
                sampler.rms[rows].zero_()
                sampler.query[rows].zero_()
                sampler.anchors[bs:bucket].zero_()
                sampler.candidate_out[bs:bucket].zero_()
            self.entries[bucket].replay()
            return self.graph_buffers.output[: bs * (self.sampler.block_size - 1)]
        buffers = self.eager_buffers
        if buffers is None or buffers.bs < bs:
            capacity = min(
                self.sampler.max_bs, max(bs, 2 * buffers.bs if buffers else 1)
            )
            self.eager_buffers = buffers = _Buffers(capacity, self.sampler)
        return self._compute(bs, buffers)
