"""DSpark prefix allocation within a hardware-profiled BS8 graph budget."""

import json
from pathlib import Path

import torch
import triton
import triton.language as tl

from sglang_dspine.verification.ragged import RaggedVerifyCaps


def conditional_table(data):
    result = []
    for successes, counts in zip(data["success"], data["count"]):
        prior = (sum(successes) + 8 * 0.85) / (sum(counts) + 8)
        result.append(
            [
                min(0.999, max(0.05, (success + 12 * prior) / (count + 12)))
                for success, count in zip(successes, counts)
            ]
        )
    return result


@triton.jit
def _indices(
    Q, Out, BS: tl.constexpr, T: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr
):
    token = tl.arange(0, BLOCK)
    row = tl.full((BLOCK,), 0, tl.int32)
    for b in tl.static_range(BS):
        row = tl.where(token >= tl.load(Q + b + 1), b + 1, row)
    start = tl.load(Q + tl.minimum(row, BS - 1))
    tl.store(Out + token, row * T + token - start, token < N)


def device_plan(raw, probabilities, depth, budget, cfg):
    from sglang.kernels.ops.speculative.dspark.dspark_schedule import (
        ScheduleVerifyLensTopk,
    )
    from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

    bins = (raw * 16).to(torch.int64).clamp_(0, 15)
    conditional = probabilities[depth, bins]
    lens = ScheduleVerifyLensTopk.execute(
        confidence=conditional, budget=budget - 8, cfg=cfg
    )
    layout = RaggedVerifyLayout._assemble_device(
        verify_lens=lens, graph_num_tokens=budget, total_verify_tokens=budget
    )
    index = torch.empty(budget, device=raw.device, dtype=torch.int64)
    _indices[(1,)](
        layout.qo_indptr_device, index, 8, 16, budget, triton.next_power_of_2(budget)
    )
    return layout, index


class RaggedSpsCaps(RaggedVerifyCaps):
    def __init__(self, worker):
        super().__init__(worker)
        from sglang.srt.speculative.dspark_components.dspark_planner import (
            DSparkScheduleConfig,
        )

        data = json.loads(Path(__file__).with_name("calibration.json").read_text())
        self.probabilities = torch.tensor(
            conditional_table(data), device=worker.device, dtype=torch.float32
        )
        self.depth = torch.arange(15, device=worker.device)
        self.cfg = DSparkScheduleConfig(
            gamma=15, min_verify_len=1, max_verify_len=16, survival_eps=0.0
        )
        self.cfg.validate()
        if (
            self.ragged_backend.needs_cpu_seq_lens
            or self.states[16][0].needs_cpu_seq_lens
        ):
            raise ValueError("DSpine SPS requires device-only FA3/GDN verify layouts")
        raw = worker._dspine_sampler.confidence_out[:8]
        raw.fill_(1.0)
        device_plan(raw, self.probabilities, self.depth, self.tokens, self.cfg)
        torch.cuda.synchronize()
        self.plan_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.plan_graph):
            self.plan_layout, self.plan_window_index = device_plan(
                raw, self.probabilities, self.depth, self.tokens, self.cfg
            )

    def prepare(self, original, verify_input, batch, target_worker, *args, **kwargs):
        if self.in_round and self.ragged_round:
            self.plan_graph.replay()
            self.layout = self.plan_layout
            self.window_index = self.plan_window_index
            self.device_only_active = True
            self.lens_cpu = self.lens_tensor_cpu = None
        return super().prepare(
            original, verify_input, batch, target_worker, *args, **kwargs
        )
