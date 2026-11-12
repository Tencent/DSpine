"""DSpark prefix allocation within a measured, per-batch token budget."""

import json
from pathlib import Path

import torch
import triton
import triton.language as tl

from sglang_dspine.verification.calibration import FORMAT, load_calibration_models
from sglang_dspine.verification.ragged import RaggedVerifyCaps, graph_tokens_for_budget


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
def _indices(Q, Out, T: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    token = tl.arange(0, BLOCK)
    start, end = tl.load(Q + row), tl.load(Q + row + 1)
    tl.store(Out + start + token, row * T + token, token < end - start)


def allocate_verify_lens(raw, probabilities, depth, budget, boundaries=None):
    """Keep the highest-survival prefixes, including one root per request."""
    bs, gamma = raw.shape
    full = gamma + 1
    if not bs <= budget <= bs * full:
        raise ValueError("SPS budget must include every root and fit the draft block")
    n_bins = probabilities.shape[-1]
    bins = (
        torch.bucketize(raw.contiguous(), boundaries) if boundaries is not None
        else (raw * n_bins).to(torch.int64).clamp_(0, n_bins - 1)
    )
    conditional = probabilities[depth, bins]
    # A stable position-major sort preserves prefix selection when survival
    # probabilities tie. Unlike pairwise ranking, this scales as O(B*T log(B*T)).
    # Parallel FP32 products can round upward; descendants must never outrank
    # their ancestors, including when scores tie at one or underflow to zero.
    survival = conditional.cumprod(dim=-1).cummin(dim=-1).values
    survival = survival.transpose(0, 1).contiguous().view(-1)
    selected = torch.argsort(survival, descending=True, stable=True)[: budget - bs]
    lens = torch.ones((bs,), device=raw.device, dtype=torch.int32)
    lens.scatter_add_(
        0, selected.remainder(bs), torch.ones_like(selected, dtype=torch.int32)
    )
    return lens


def device_plan(raw, probabilities, depth, budget, cfg, graph_tokens=None, boundaries=None):
    if cfg.min_verify_len != 1 or cfg.survival_eps != 0.0:
        raise ValueError("DSpine SPS requires one root per request and no survival cutoff")
    bs, gamma = raw.shape
    full = gamma + 1
    if graph_tokens is None:
        graph_tokens = budget
    if graph_tokens != budget:
        raise ValueError("SPS graph tier must equal the verified token budget")
    from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

    lens = allocate_verify_lens(raw, probabilities, depth, budget, boundaries)
    layout = RaggedVerifyLayout._assemble_device(
        verify_lens=lens, graph_num_tokens=graph_tokens, total_verify_tokens=budget
    )
    index = torch.empty(budget, device=raw.device, dtype=torch.int64)
    _indices[(bs,)](layout.qo_indptr_device, index, full, triton.next_power_of_2(full))
    return layout, index


class RaggedSpsCaps(RaggedVerifyCaps):
    def __init__(self, worker):
        super().__init__(worker)
        self.plans = {}
        if self.ragged_backend is None:
            return
        from sglang.srt.speculative.dspark_components.dspark_planner import (
            DSparkScheduleConfig,
        )
        from sglang.srt.runtime_context import get_spec

        path = get_spec().speculative_dspark_sps_table_path
        profile = json.loads(Path(path).read_text()) if path else {}
        calibration = profile.get("confidence_calibration")
        if calibration is None:
            data = json.loads(Path(__file__).with_name("calibration.json").read_text())
            if data.get("format") == FORMAT:
                calibration = data
        if calibration is None:
            legacy = torch.tensor(
                conditional_table(data), device=worker.device, dtype=torch.float32
            )
            self.probabilities = {"greedy": legacy, "sampled": legacy}
            self.boundaries = {"greedy": None, "sampled": None}
        else:
            models = load_calibration_models(calibration, self.full - 1)
            self.probabilities = {
                mode: model.probabilities.to(worker.device) for mode, model in models.items()
            }
            self.boundaries = {
                mode: model.boundaries.to(worker.device) if model.boundaries.numel() else None
                for mode, model in models.items()
            }
        self.depth = torch.arange(self.full - 1, device=worker.device)
        self.cfg = DSparkScheduleConfig(
            gamma=self.full - 1,
            min_verify_len=1,
            max_verify_len=self.full,
            survival_eps=0.0,
        )
        self.cfg.validate()
        # Only the explicitly profiled batches get tiny planner graphs. The
        # target has one shared ragged graph runner, not a model capture per
        # batch/budget pair. Unprofiled batches keep full-block verification.
        if self.states[self.full][1] is not None:
            worker._dspine_sampler.confidence_out.fill_(1.0)
            for family, probabilities in self.probabilities.items():
                for bs, budget in self.policy_for_mode(family).token_budgets.items():
                    if budget == bs * self.full or bs > worker._dspine_sampler.max_bs:
                        continue
                    tier = graph_tokens_for_budget(self.ragged_graph, budget, bs)
                    if tier is None:
                        continue
                    raw = worker._dspine_sampler.confidence_out[:bs]
                    device_plan(raw, probabilities, self.depth, budget, self.cfg, tier, self.boundaries[family])
                    torch.cuda.synchronize(worker.device)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        layout, index = device_plan(
                            raw, probabilities, self.depth, budget, self.cfg, tier, self.boundaries[family]
                        )
                    self.plans[family, bs, budget, tier] = graph, layout, index

    def prepare(self, original, verify_input, batch, target_worker, *args, **kwargs):
        if self.in_round and self.ragged_round:
            family = "sampled" if self.sampled_round else "greedy"
            plan = self.plans.get((family, self.batch_size, self.tokens, self.graph_tokens))
            if plan is None:
                self.layout, self.window_index = device_plan(
                    self.worker._dspine_sampler.confidence_out[: self.batch_size],
                    self.probabilities[family],
                    self.depth,
                    self.tokens,
                    self.cfg,
                    self.graph_tokens,
                    self.boundaries[family],
                )
            else:
                graph, self.layout, self.window_index = plan
                graph.replay()
            self.device_only_active = True
            self.lens_cpu = self.lens_tensor_cpu = None
        return super().prepare(
            original, verify_input, batch, target_worker, *args, **kwargs
        )
