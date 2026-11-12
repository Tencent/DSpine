"""Verify policy loading, measured selection and device-side prefix allocation."""

import importlib.util
import json
import socket
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang_dspine.verification.policy import SpsPolicy


def tuner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "tune_sps.py"
    spec = importlib.util.spec_from_file_location("tune_sps", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unprofiled_engine_verifies_all():
    policy = SpsPolicy.load(None, {})
    assert policy == SpsPolicy()
    for bs in (1, 3, 8, 9, 17, 64, 257):
        assert policy.width(bs) == 16
        assert policy.budget(bs) == bs * 16


@pytest.mark.parametrize(
    "widths,budgets",
    [
        ({0: 12}, {}),
        ({1: True}, {}),
        ({9: 17}, {}),
        ({}, {9: 8}),
        ({}, {9: 145}),
        ({9: 12}, {9: 95}),
        ({}, {True: 16}),
    ],
)
def test_invalid_policy_cannot_reach_graph_capture(widths, budgets):
    with pytest.raises(ValueError):
        SpsPolicy(widths, budgets)


def test_budgets_need_not_be_batch_or_graph_tier_multiples():
    policy = SpsPolicy({3: 7}, {9: 95, 17: 201, 65: 65})
    assert policy.width(3) == 7 and policy.budget(9) == 95
    assert policy.budget(17) == 201 and policy.budget(65) == 65
    assert policy.budget(10) == 160


def test_profile_is_bound_to_hardware_and_uses_its_measured_widths(tmp_path):
    hardware = {
        "name": "NVIDIA H200",
        "capability": [9, 0],
        "memory_bytes": 140000000000,
    }
    policy = SpsPolicy({3: 12}, {9: 95, 32: 256})
    path = tmp_path / "sps.json"
    path.write_text(
        json.dumps(
            {
                "format": "dspine-sps-v2",
                "hardware": hardware,
                "policy": policy.to_dict(),
            }
        )
    )
    assert SpsPolicy.load(path, hardware).to_dict() == policy.to_dict()
    with pytest.raises(ValueError, match="hardware"):
        SpsPolicy.load(path, {**hardware, "name": "NVIDIA H100"})
    path.write_text('{"sample_steps_per_sec": [100]}')
    with pytest.raises(ValueError, match="DSpine SPS profile"):
        SpsPolicy.load(path, hardware)


def test_legacy_profile_only_profiles_its_original_batches(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({
        "format": "dspine-sps-v1", "hardware": {},
        "policy": {"uniform_widths": [16, 16, 12, 12, 8, 8, 8], "batch8_tokens": 64},
    }))
    policy = SpsPolicy.load(path, {})
    assert policy.width(3) == 12 and policy.budget(8) == 64
    assert policy.width(9) == 16 and policy.budget(9) == 144


def test_selects_wall_throughput_independently_for_each_batch_size():
    module = tuner()
    rows = [
        {"batch_size": b, "width": w, "token_budget": 64 if (b, w) == (17, 4) else b * w, "tokens_per_second": s}
        for b, w, s in [
            (1, 12, 90),
            (1, 16, 100),
            (4, 12, 500),
            (4, 16, 400),
            (8, 8, 800),
            (8, 12, 850),
            (8, 16, 850),
            (17, 4, 1300),
            (17, 16, 900),
        ]
    ]
    assert module.select_policy(rows).to_dict() == {
        "verify_widths": {},
        "token_budgets": {"1": 16, "4": 48, "8": 128, "17": 64},
    }


def test_tuner_maps_arbitrary_batches_to_actual_native_token_tiers():
    module = tuner()
    info = {"speculative_num_draft_tokens": 16, "cuda_graph_config": {"decode": {"backend": "full", "bs": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20]}}}
    grid = module.native_token_grid(info)
    assert grid == [16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 288, 320]
    assert module.candidate_budgets([3, 7, 17], 12, grid) == {3: 32, 7: 80, 17: 192}
    assert module.candidate_budgets([3, 7, 17], 8, grid) == {3: 32, 7: 64, 17: 128}
    assert module.candidate_budgets([3, 7, 17], 16, grid) == {3: 48, 7: 112, 17: 272}
    assert module.candidate_budgets([17], 4, [16]) == {17: 272}
    assert module.candidate_budgets([17], 12, [16, 32, 64, 128, 256]) == {17: 272}
    # There is no interpolation to unprofiled batch sizes.
    assert SpsPolicy(token_budgets=module.candidate_budgets([3], 12, grid)).budget(7) == 112


@pytest.mark.parametrize("info", [
    {},
    {"speculative_num_draft_tokens": 16},
    {"speculative_num_draft_tokens": 16, "cuda_graph_config": {"decode": {"backend": "disabled", "bs": [1, 2, 4]}}},
    {"speculative_num_draft_tokens": 16, "disable_decode_cuda_graph": True, "cuda_graph_config": {"decode": {"backend": "full", "bs": [1, 2, 4]}}},
])
def test_tuner_without_a_captured_grid_keeps_full_block(info):
    module = tuner()
    assert module.native_token_grid(info) == []
    assert module.candidate_budgets([3, 7, 17], 12, module.native_token_grid(info)) == {3: 48, 7: 112, 17: 272}


def test_accepts_mixed_greedy_and_sampled_workload(tmp_path):
    path = tmp_path / "prefixes.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"input_ids": [1, 2], "sampling_params": {"temperature": t}})
            for t in [0, 1]
        )
    )
    assert len(tuner().load_prefixes(path)) == 2


def test_sequential_candidates_can_reuse_the_same_endpoint(tmp_path):
    module = tuner()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    program = """
import http.server, sys
class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Length', '2')
        self.end_headers()
        self.wfile.write(b'{}')
        self.close_connection = True
http.server.HTTPServer(('127.0.0.1', int(sys.argv[1])), Handler).serve_forever()
"""
    command = [sys.executable, "-u", "-c", program, str(port)]
    for i in range(2):
        with module.server(
            command, tmp_path / "sps.json", tmp_path / str(i), port, 10
        ) as url:
            assert module.request(url + "/model_info") == {}


def test_policy_routes_real_decode_batches_and_keeps_ineligible_full_width():
    from sglang_dspine.verification.caps import VerifyCaps

    controller = object.__new__(VerifyCaps)
    controller.policy = SpsPolicy({2: 12, 9: 8, 17: 6}, {8: 64})
    controller.full = 16
    controller.sampled_dynamic = True
    controller.counts = Counter()
    for bs in (1, 2, 3, 8, 9, 17, 65):
        batch = SimpleNamespace(
            batch_size=lambda: bs,
            sampling_info=SimpleNamespace(is_all_greedy=True),
            forward_mode=SimpleNamespace(is_decode=lambda: True),
            has_grammar=False,
            return_logprob=False,
        )
        controller.begin(batch)
        assert controller.active == controller.policy.width(bs)
        batch.sampling_info = SimpleNamespace(is_all_greedy=False, is_any_greedy=False)
        controller.begin(batch)
        assert controller.active == controller.policy.width(bs) and controller.in_round
        batch.sampling_info.is_any_greedy = True
        controller.begin(batch)
        assert controller.active == 16 and not controller.in_round
        batch.has_grammar = True
        controller.begin(batch)
        assert controller.active == 16 and not controller.in_round


@pytest.mark.parametrize("bs,average_width", [(1, 1), (3, 4), (8, 12), (9, 7), (17, 8), (65, 3)])
def test_sps_device_plan_captures_and_maps_exactly_the_budget(bs, average_width):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from sglang.srt.speculative.dspark_components.dspark_planner import (
        DSparkScheduleConfig,
    )

    from sglang_dspine.verification.sps import device_plan

    torch.manual_seed(52)
    budget = bs * average_width + (bs > 1)
    raw = torch.rand((bs, 15), device="cuda")
    table = torch.linspace(0.1, 0.99, 16, device="cuda").expand(15, 16).contiguous()
    depth = torch.arange(15, device="cuda")
    cfg = DSparkScheduleConfig(
        gamma=15, min_verify_len=1, max_verify_len=16, survival_eps=0.0
    )
    graph_tokens = budget
    device_plan(raw, table, depth, budget, cfg, graph_tokens)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        layout, indices = device_plan(raw, table, depth, budget, cfg, graph_tokens)
    for _ in range(2):
        raw.uniform_()
        graph.replay()
        lens = layout.verify_lens.cpu().tolist()
        assert layout.graph_num_tokens == graph_tokens
        assert layout.total_verify_tokens == budget
        assert indices.numel() == budget
        assert all(1 <= n <= 16 for n in lens) and sum(lens) == budget
        assert indices.cpu().tolist() == [
            b * 16 + i for b, n in enumerate(lens) for i in range(n)
        ]


def test_ragged_policy_uses_actual_batch_and_never_routes_mixed_sampling():
    from sglang_dspine.verification.ragged import RaggedVerifyCaps

    controller = object.__new__(RaggedVerifyCaps)
    controller.policy = SpsPolicy(token_budgets={3: 19, 9: 95, 17: 201})
    controller.full = 16
    controller.sampled_dynamic = True
    controller.counts = Counter()
    controller.ragged_backend = object()
    controller.ragged_graph = None
    for bs in (3, 9, 17, 32):
        batch = SimpleNamespace(
            batch_size=lambda: bs,
            sampling_info=SimpleNamespace(is_all_greedy=False, is_any_greedy=False),
            forward_mode=SimpleNamespace(is_decode=lambda: True),
            has_grammar=False, return_logprob=False,
        )
        controller.begin(batch)
        assert controller.ragged_round == (bs in (3, 9, 17))
        assert controller.tokens == controller.policy.budget(bs)
        batch.sampling_info.is_any_greedy = True
        controller.begin(batch)
        assert not controller.ragged_round and controller.active == 16


def test_ragged_tier_matches_captured_tokens_and_request_slots():
    from sglang_dspine.verification.ragged import graph_tokens_for_budget

    graph = SimpleNamespace(
        capture_num_tokens=[16, 32, 48, 64, 96, 128, 208],
        _ragged_capture_slots=lambda tier: min(tier, 32),
    )
    assert graph_tokens_for_budget(graph, 95, 9) is None
    assert graph_tokens_for_budget(graph, 201, 17) is None
    assert graph_tokens_for_budget(graph, 96, 9) == 96
    assert graph_tokens_for_budget(graph, 208, 17) == 208
    assert graph_tokens_for_budget(graph, 209, 17) is None
    assert graph_tokens_for_budget(graph, 95, 33) is None
    assert graph_tokens_for_budget(None, 95, 9) is None
    graph._ragged_capture_slots = lambda tier: tier // 16
    assert graph_tokens_for_budget(graph, 32, 3) is None
    assert graph_tokens_for_budget(graph, 48, 3) == 48


def test_planner_rejects_padding_a_budget_to_another_graph_tier():
    import torch

    from sglang_dspine.verification.sps import device_plan

    with pytest.raises(ValueError, match="must equal"):
        device_plan(
            torch.ones(1, 15), torch.ones(15, 16), torch.arange(15),
            8, SimpleNamespace(min_verify_len=1, survival_eps=0.0), graph_tokens=16,
        )


@pytest.mark.parametrize("bs,budget", [(1, 8), (3, 19), (9, 95), (17, 201)])
@pytest.mark.parametrize("has_exact_graph", [False, True])
def test_ragged_prepare_uses_exact_budget_without_padding(bs, budget, has_exact_graph):
    import torch

    from sglang_dspine.verification.ragged import RaggedVerifyCaps

    controller = object.__new__(RaggedVerifyCaps)
    controller.full, controller.batch_size, controller.tokens = 16, bs, budget
    controller.in_round = controller.ragged_round = True
    controller.compact_round = False
    controller.device_only_active = True
    controller.layout = SimpleNamespace(graph_num_tokens=budget)
    controller.window_index = torch.arange(budget)
    controller.ragged_backend = SimpleNamespace(needs_cpu_seq_lens=False)
    controller.round_graph = object() if has_exact_graph else None
    controller.runner = SimpleNamespace()
    controller.executed_counts = Counter()
    controller.lens_cpu = None
    original_ids = torch.arange(bs * 16, dtype=torch.int64)
    verify_input = SimpleNamespace(
        draft_token=original_ids,
        positions=original_ids + 100,
        custom_mask=None,
        ragged_verify_layout=None,
        kv_loc_cols=None,
        live_seq_lens_cpu=torch.zeros(bs, dtype=torch.int64),
    )
    batch = SimpleNamespace(
        input_ids=original_ids,
        out_cache_loc=original_ids + 200,
        seq_lens_cpu=torch.full((bs,), 16),
        seq_lens_sum=bs * 16,
    )

    def prepare(vi, b, target_worker):
        b.input_ids = vi.draft_token
        assert vi.draft_token.numel() == vi.positions.numel() == budget
        assert b.out_cache_loc.numel() == vi.kv_loc_cols.numel() == budget
        assert controller.window_index.numel() == budget
        assert vi.ragged_verify_layout.graph_num_tokens == budget
        assert controller.runner.decode_cuda_graph_runner is controller.round_graph

    controller.prepare(prepare, verify_input, batch, object())
    rows = budget
    logits = torch.arange(rows * 2, dtype=torch.float32).view(rows, 2)
    hidden = torch.arange(rows * 3, dtype=torch.float32).view(rows, 3)
    result = SimpleNamespace(
        logits_output=SimpleNamespace(next_token_logits=logits, hidden_states=hidden),
        can_run_cuda_graph=has_exact_graph,
    )
    result = controller.target(lambda *a, **k: result, object(), is_verify=True)
    assert controller.executed_tokens == budget
    assert result.logits_output.next_token_logits.shape == (bs * 16, 2)
    assert torch.equal(result.logits_output.next_token_logits[:budget], logits[:budget])
    assert torch.equal(result.logits_output.hidden_states[:budget], hidden[:budget])
    assert verify_input.draft_token is original_ids
    assert batch.input_ids is original_ids


@pytest.mark.parametrize("bs", [1, 3, 9, 17, 65])
def test_allocation_matches_global_survival_order_and_preserves_prefixes(bs):
    import torch

    from sglang_dspine.verification.sps import allocate_verify_lens

    generator = torch.Generator().manual_seed(714)
    raw = torch.rand((bs, 15), generator=generator)
    # Equal confidence creates ties across rows and depths. They must never
    # select a descendant without all of its preceding draft positions.
    raw[::2] = 1
    table = torch.linspace(0.1, 1.0, 16).expand(15, 16)
    conditional = table[torch.arange(15), (raw * 16).long().clamp(0, 15)]
    survival = conditional.cumprod(-1)
    order = sorted(
        ((b, d) for b in range(bs) for d in range(15)),
        key=lambda pair: (-float(survival[pair]), pair[1], pair[0]),
    )
    for budget in (bs, bs * 7 + (bs > 1), bs * 16):
        lens = allocate_verify_lens(raw, table, torch.arange(15), budget)
        selected = set(order[:budget - bs])
        expected = [1 + sum((b, d) in selected for d in range(15)) for b in range(bs)]
        assert lens.tolist() == expected and int(lens.sum()) == budget
        assert all((b, d) in selected for b, n in enumerate(expected) for d in range(n - 1))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fp32_extreme_survival_values_preserve_ancestor_order(device):
    import torch

    from sglang_dspine.verification.sps import allocate_verify_lens

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    bs, gamma = 4, 15
    conditional = torch.tensor([
        [1.0] * gamma,
        [1.0, 1.0 - 2**-24, 1.0 - 2**-23] * 5,
        [1e-30, 1.0, 1e-30] + [1.0] * 12,
        [1.0, torch.finfo(torch.float32).tiny, 1.0, 1e-10] + [1.0] * 11,
    ], dtype=torch.float32, device=device)
    # Four distinct bins let real table lookup reproduce the extreme rows.
    table = conditional.T.contiguous()
    raw = torch.tensor([0.125, 0.375, 0.625, 0.875], device=device)[:, None].expand(bs, gamma)
    depth = torch.arange(gamma, device=device)
    survival = conditional.cumprod(-1).cummin(-1).values
    assert torch.isfinite(survival).all() and (survival[:, 1:] <= survival[:, :-1]).all()
    values = survival.cpu().tolist()
    order = sorted(((b, d) for b in range(bs) for d in range(gamma)), key=lambda pair: (-values[pair[0]][pair[1]], pair[1], pair[0]))
    previous = [1] * bs
    for budget in range(bs, bs * (gamma + 1) + 1):
        lens = allocate_verify_lens(raw, table, depth, budget).cpu().tolist()
        selected = set(order[:budget - bs])
        expected = [1 + sum((b, d) in selected for d in range(gamma)) for b in range(bs)]
        assert lens == expected and sum(lens) == budget
        assert all(1 <= n <= gamma + 1 and n >= old for n, old in zip(lens, previous))
        assert all((b, d) in selected for b, n in enumerate(lens) for d in range(n - 1))
        previous = lens


def test_logit_lookup_bucketize_device_plan_captures_and_replays():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from sglang.srt.speculative.dspark_components.dspark_planner import DSparkScheduleConfig
    from sglang_dspine.verification.sps import allocate_verify_lens, device_plan

    logits = torch.linspace(-13.815509557963773, 13.815509557963773, 1024, device="cuda", dtype=torch.float64)
    boundaries = ((logits[1:] + logits[:-1]) / 2).sigmoid()
    table = (0.216 * logits + 0.3).sigmoid().float().repeat(15, 1)
    raw = torch.ones((3, 15), device="cuda", dtype=torch.float32)
    depth = torch.arange(15, device="cuda")
    cfg = DSparkScheduleConfig(gamma=15, min_verify_len=1, max_verify_len=16, survival_eps=0.0)
    budget = 17
    device_plan(raw, table, depth, budget, cfg, boundaries=boundaries)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        layout, indices = device_plan(raw, table, depth, budget, cfg, boundaries=boundaries)
    for tail in (0.0, 1.0, 1 - 1e-6):
        raw.uniform_()
        raw[0].fill_(tail)
        graph.replay()
        expected = allocate_verify_lens(raw, table, depth, budget, boundaries)
        torch.testing.assert_close(layout.verify_lens, expected)
        lens = expected.cpu().tolist()
        assert sum(lens) == budget
        assert indices.cpu().tolist() == [b * 16 + d for b, n in enumerate(lens) for d in range(n)]
