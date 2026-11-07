"""One immutable full-vocabulary weight cache and a q graph for each batch size."""

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


class ProposalGraphs:
    def __init__(self, sampler):
        self.sampler = sampler
        self.entries = {}
        self.inputs = {}
        self.workspaces = {}
        self.base = {}
        self.stream = None
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
        for bs in range(1, 9):
            n = bs * 15
            self.inputs[bs] = (
                sampler.raw[: bs * 16],
                sampler.rms[: bs * 16],
                sampler.query[: bs * 16],
                sampler.anchors[:bs],
                sampler.candidate_out[:bs].view(n, 16),
            )
            self.base[bs] = torch.empty(
                (n, 16), device=sampler.raw.device, dtype=torch.float64
            )
            self.workspaces[bs] = Workspace.allocate(
                n, 16, 5120, 512, sampler.raw.device
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

    def _compute(self, bs, output):
        raw, rms, query, anchors, candidates = self.inputs[bs]
        return factor_table(
            raw,
            rms,
            query,
            anchors,
            candidates,
            self.weights,
            self.sampler.norm_eps,
            16,
            output,
            self.base[bs],
            self.workspaces[bs],
        )

    def lookup(self, bs):
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
        graph, output, _ = self.entries[bs]
        graph.replay()
        return output
