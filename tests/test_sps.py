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
    assert SpsPolicy.load(None, {}) == SpsPolicy((16,) * 7, 128)


@pytest.mark.parametrize(
    "widths,tokens",
    [
        ((16,) * 6, 96),
        ((True,) * 7, 96),
        ((17,) * 7, 96),
        ((12,) * 7, 95),
        ((12,) * 7, 144),
    ],
)
def test_invalid_policy_cannot_reach_graph_capture(widths, tokens):
    with pytest.raises(ValueError):
        SpsPolicy(widths, tokens)


def test_profile_is_bound_to_hardware_and_uses_its_measured_widths(tmp_path):
    hardware = {
        "name": "NVIDIA H200",
        "capability": [9, 0],
        "memory_bytes": 140000000000,
    }
    policy = SpsPolicy((16, 16, 12, 12, 8, 8, 8), 64)
    path = tmp_path / "sps.json"
    path.write_text(
        json.dumps(
            {
                "format": "dspine-sps-v1",
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


def test_selects_wall_throughput_independently_for_each_batch_size():
    module = tuner()
    rows = [
        {"batch_size": b, "width": w, "tokens_per_second": s}
        for b, w, s in [
            (1, 12, 90),
            (1, 16, 100),
            (4, 12, 500),
            (4, 16, 400),
            (8, 8, 800),
            (8, 12, 850),
            (8, 16, 850),
        ]
    ]
    assert module.select_policy(rows).to_dict() == {
        "uniform_widths": [16, 16, 16, 12, 16, 16, 16],
        "batch8_tokens": 128,
    }


def test_rejects_mixed_greedy_and_sampled_workload(tmp_path):
    path = tmp_path / "prefixes.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"input_ids": [1, 2], "sampling_params": {"temperature": t}})
            for t in [0, 1]
        )
    )
    with pytest.raises(ValueError, match="separate runs"):
        tuner().load_prefixes(path)


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
    controller.policy = SpsPolicy((16, 12, 8, 8, 8, 6, 6), 64)
    controller.full = 16
    controller.sampled_dynamic = True
    controller.counts = Counter()
    for bs in range(1, 9):
        batch = SimpleNamespace(
            batch_size=lambda: bs,
            sampling_info=SimpleNamespace(is_all_greedy=True),
            forward_mode=SimpleNamespace(is_decode=lambda: True),
            has_grammar=False,
            return_logprob=False,
        )
        controller.begin(batch)
        assert controller.active == (
            controller.policy.uniform_widths[bs - 1] if bs < 8 else 16
        )
        batch.has_grammar = True
        controller.begin(batch)
        assert controller.active == 16 and not controller.in_round


@pytest.mark.parametrize("budget", [16, 32, 64, 96, 112, 128])
def test_sps_device_plan_captures_and_maps_exactly_the_budget(budget):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from sglang.srt.speculative.dspark_components.dspark_planner import (
        DSparkScheduleConfig,
    )

    from sglang_dspine.verification.sps import device_plan

    torch.manual_seed(52)
    raw = torch.rand((8, 15), device="cuda")
    table = torch.linspace(0.1, 0.99, 16, device="cuda").expand(15, 16).contiguous()
    depth = torch.arange(15, device="cuda")
    cfg = DSparkScheduleConfig(
        gamma=15, min_verify_len=1, max_verify_len=16, survival_eps=0.0
    )
    device_plan(raw, table, depth, budget, cfg)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        layout, indices = device_plan(raw, table, depth, budget, cfg)
    for _ in range(2):
        raw.uniform_()
        graph.replay()
        lens = layout.verify_lens.cpu().tolist()
        assert all(1 <= n <= 16 for n in lens) and sum(lens) == budget
        assert indices.cpu().tolist() == [
            b * 16 + i for b, n in enumerate(lens) for i in range(n)
        ]
