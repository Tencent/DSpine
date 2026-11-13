"""CPU proofs that mode switches reuse the graph bound to the correct table."""

import contextlib
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang_dspine.verification.calibration import FORMAT, load_calibration_tables
from sglang_dspine.verification.policy import SpsPolicy
from sglang_dspine.verification import sps


def artifact(grid="linear"):
    bins = 256
    greedy = ((torch.arange(bins) + 0.5) / bins).repeat(15, 1)
    sampled = torch.full((15, bins), 0.5)
    boundaries = {}
    if grid == "logit":
        logits = torch.linspace(-13.815509557963773, 13.815509557963773, bins, dtype=torch.float64)
        representatives = logits.sigmoid().float()
        representatives[0], representatives[-1] = 0, 1
        greedy = representatives.repeat(15, 1)
        boundaries = {"boundaries": ((logits[1:] + logits[:-1]) / 2).sigmoid().tolist()}
    key = str(bins) if grid == "linear" else f"logit{bins}"
    return {
        "format": FORMAT, "gamma": 15, "lookup_bins": bins, "lookup_grid": grid,
        "families": {
            mode: {"lookup_tables": {key: {"kind": "table", "gamma": 15, "probabilities": table.tolist(), "monotonic": True, **boundaries}}}
            for mode, table in (("greedy", greedy), ("sampled", sampled))
        },
    }


def controller(monkeypatch, tmp_path, calibration, capture, packaged=None, mode_policies=None):
    path = tmp_path / "sps.json"
    path.write_text(json.dumps({"confidence_calibration": calibration} if calibration else {}))
    if packaged is not None:
        packaged_path = Path(sps.__file__).with_name("calibration.json")
        original_read = Path.read_text

        def read_text(path, *args, **kwargs):
            return json.dumps(packaged) if path == packaged_path else original_read(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", read_text)

    def parent_init(self, worker):
        self.worker, self.full = worker, 16
        self.ragged_backend = object()
        self.policy = SpsPolicy(token_budgets={2: 4})
        if mode_policies is not None:
            self.default_policy, self.mode_policies = self.policy, mode_policies
        self.states = {16: (None, object() if capture else None)}
        tiers = {4, *(budget for policy in (mode_policies or {}).values() for budget in policy.token_budgets.values())}
        self.ragged_graph = SimpleNamespace(capture_num_tokens=sorted(tiers), _ragged_capture_slots=lambda tier: 2) if capture else None

    class Config(SimpleNamespace):
        def validate(self):
            assert self.gamma == 15 and self.min_verify_len == 1

    monkeypatch.setattr(sps.RaggedVerifyCaps, "__init__", parent_init)
    monkeypatch.setattr(sps.RaggedVerifyCaps, "prepare", lambda self, original, *args, **kwargs: original(*args, **kwargs))
    monkeypatch.setitem(sys.modules, "sglang.srt.runtime_context", SimpleNamespace(get_spec=lambda: SimpleNamespace(speculative_dspark_sps_table_path=path)))
    monkeypatch.setitem(sys.modules, "sglang.srt.speculative.dspark_components.dspark_planner", SimpleNamespace(DSparkScheduleConfig=Config))
    active, created, planned = [], [], []

    class Graph:
        def __init__(self):
            self.replays = 0
            created.append(self)

        def replay(self):
            self.replays += 1
            self.refresh()

    @contextlib.contextmanager
    def graph_context(graph):
        active.append(graph)
        try:
            yield
        finally:
            active.pop()

    def plan(raw, table, depth, budget, cfg, tier, boundaries=None):
        planned.append(table)
        lens = sps.allocate_verify_lens(raw, table, depth, budget, boundaries)
        layout = SimpleNamespace(verify_lens=lens, probabilities=table)
        if active:
            active[-1].refresh = lambda: lens.copy_(sps.allocate_verify_lens(raw, table, depth, budget, boundaries))
        return layout, torch.empty(budget, dtype=torch.int64)

    monkeypatch.setattr(sps, "device_plan", plan)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(torch.cuda, "graph", graph_context)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    worker = SimpleNamespace(device="cpu", _dspine_sampler=SimpleNamespace(confidence_out=torch.empty(2, 15), max_bs=2))
    result = sps.RaggedSpsCaps(worker)
    result.in_round = result.ragged_round = True
    result.batch_size, result.tokens = 2, 4
    result.graph_tokens = 4 if capture else None
    worker._dspine_sampler.confidence_out[0].fill_(0.99)
    worker._dspine_sampler.confidence_out[1].fill_(0.1)
    return result, created, planned


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("grid", ["linear", "logit"])
def test_mode_switch_uses_correct_table_and_reuses_bound_planner_graph(monkeypatch, tmp_path, capture, grid):
    caps, graphs, planned = controller(monkeypatch, tmp_path, artifact(grid), capture)
    original = lambda *args: "prepared"
    for sampled, expected in ((False, [3, 1]), (True, [2, 2]), (False, [3, 1])):
        caps.sampled_round = sampled
        assert caps.prepare(original, object(), object(), object()) == "prepared"
        assert caps.layout.verify_lens.tolist() == expected
        assert caps.layout.probabilities is caps.probabilities["sampled" if sampled else "greedy"]
    if capture:
        assert len(graphs) == 2 and [g.replays for g in graphs] == [2, 1]
        assert set(caps.plans) == {("greedy", 2, 4, 4), ("sampled", 2, 4, 4)}
        # Warmup + capture only: switching/replay performs no new planning/copy.
        assert len(planned) == 4
    else:
        assert len(planned) == 3 and not caps.plans
    assert caps.device_only_active and caps.lens_cpu is caps.lens_tensor_cpu is None


@pytest.mark.parametrize("greedy_budget", [6, 32])
def test_mode_specific_budgets_capture_only_the_required_family_plans(monkeypatch, tmp_path, greedy_budget):
    policies = {"greedy": SpsPolicy(token_budgets={2: greedy_budget}), "sampled": SpsPolicy(token_budgets={2: 4})}
    caps, graphs, _ = controller(monkeypatch, tmp_path, artifact("logit"), True, mode_policies=policies)
    expected_keys = {("sampled", 2, 4, 4)}
    if greedy_budget < 32:
        expected_keys.add(("greedy", 2, 6, 6))
    assert set(caps.plans) == expected_keys and len(graphs) == len(expected_keys)
    for mode in ("greedy", "sampled", "greedy"):
        caps.sampled_round = mode == "sampled"
        caps.policy = caps.policy_for_mode(mode)
        caps.tokens = caps.policy.budget(2)
        caps.ragged_round = caps.tokens < 32
        caps.graph_tokens = caps.tokens
        assert caps.prepare(lambda *args: "prepared", object(), object(), object()) == "prepared"
        if caps.ragged_round:
            assert int(caps.layout.verify_lens.sum()) == caps.tokens
            assert caps.layout.probabilities is caps.probabilities[mode]


def test_legacy_profile_preserves_original_table_for_both_modes(monkeypatch, tmp_path):
    old = {"success": [list(range(16))] * 15, "count": [[16] * 16] * 15}
    caps, _, _ = controller(monkeypatch, tmp_path, None, False, packaged=old)
    expected = torch.tensor(sps.conditional_table(old))
    assert caps.probabilities["greedy"] is caps.probabilities["sampled"]
    assert caps.probabilities["greedy"].shape == (15, 16)
    torch.testing.assert_close(caps.probabilities["greedy"], expected)


def test_modern_packaged_default_loads_both_mode_tables(monkeypatch, tmp_path):
    calibration = artifact("logit")
    caps, _, _ = controller(monkeypatch, tmp_path, None, False, packaged=calibration)
    expected = load_calibration_tables(calibration)
    for mode in ("greedy", "sampled"):
        torch.testing.assert_close(caps.probabilities[mode], expected[mode])
    assert caps.probabilities["greedy"] is not caps.probabilities["sampled"]
    assert caps.boundaries["sampled"].shape == (255,)


def test_real_packaged_default_loads_without_embedded_calibration(monkeypatch, tmp_path):
    packaged = json.loads(Path(sps.__file__).with_name("calibration.json").read_text())
    caps, _, _ = controller(monkeypatch, tmp_path, None, False)
    assert packaged["format"] == FORMAT
    expected = load_calibration_tables(packaged)
    for mode in ("greedy", "sampled"):
        torch.testing.assert_close(caps.probabilities[mode], expected[mode])
        assert caps.boundaries[mode].shape == (packaged["lookup_bins"] - 1,)


def test_calibration_table_depth_and_resolution_validated_before_serving():
    value = artifact()
    value["lookup_bins"] = 1024
    with pytest.raises(KeyError):
        load_calibration_tables(value)
    value = artifact()
    value["gamma"] = 14
    with pytest.raises(ValueError, match="gamma"):
        load_calibration_tables(value)


@pytest.mark.parametrize("source", ["pre_token_max_q_actual_sampled_chain", "greedy_chain_top1_top16_softmax"])
def test_runtime_rejects_recorded_obsolete_sampled_feature(source):
    value = artifact("logit")
    value["confidence_sources"] = {"greedy": "greedy_chain_top1_top16_softmax", "sampled": source}
    with pytest.raises(ValueError, match="recollect and refit"):
        load_calibration_tables(value)


def test_runtime_accepts_recorded_pretruncation_sampled_feature():
    value = artifact("logit")
    value["confidence_sources"] = {"greedy": "greedy_chain_top1_top16_softmax", "sampled": "pre_token_max_probability_actual_sampled_chain"}
    tables = load_calibration_tables(value)
    assert set(tables) == {"greedy", "sampled"}
    assert tables["sampled"].shape == (15, 256)


@pytest.mark.parametrize("batch_sizes", [[2], [3, 7, 17]])
@pytest.mark.parametrize("graph_backend", ["full", "disabled"])
def test_tuner_embeds_same_frozen_artifact_in_trials_and_selected_profile(monkeypatch, tmp_path, batch_sizes, graph_backend):
    path = Path(__file__).resolve().parents[1] / "scripts" / "tune_sps.py"
    spec = importlib.util.spec_from_file_location("tune_sps_calibration", path)
    tuner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tuner)
    calibration = artifact()
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(calibration))
    prefixes = tmp_path / "prefixes.jsonl"
    prefixes.write_text('\n'.join(json.dumps({"input_ids": [i + 1], "sampling_params": {"temperature": 0.7}}) for i in range(max(batch_sizes))))
    output = tmp_path / "selected.json"
    observed = []

    @contextlib.contextmanager
    def server(command, profile, folder, port, timeout):
        observed.append(json.loads(profile.read_text()))
        yield "http://cpu-mock"

    def measure(url, rows, sizes, repeats, tokens, raw_path, request_timeout=None):
        assert observed[-1]["confidence_calibration"] == calibration
        assert request_timeout is None
        result = []
        for bs in sizes:
            rounds = []
            for repeat in range(repeats):
                jobs = tuner.queue_payloads(rows, bs, repeat, max(len(rows), 4 * bs), tokens)
                digest = hashlib.sha256(json.dumps([job["payload"] for job in jobs],
                    sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                rounds.append({"repeat": repeat, "requests": len(jobs), "status": "PASS", "jobs_sha256": digest})
            result.append({"batch_size": bs,
                "tokens_per_second": observed[-1]["mode_policies"]["sampled"]["token_budgets"][str(bs)],
                "rounds": rounds})
        return result

    monkeypatch.setattr(tuner, "hardware_identity", lambda device: {})
    monkeypatch.setattr(tuner, "request", lambda url: {"speculative_num_draft_tokens": 16, "cuda_graph_config": {"decode": {"backend": graph_backend, "bs": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20]}}})
    monkeypatch.setattr(tuner, "server", server)
    monkeypatch.setattr(tuner, "measure", measure)
    monkeypatch.setattr(sys, "argv", [str(path), "--requests", str(prefixes), "--output", str(output), "--confidence-calibration", str(calibration_path), "--batch-sizes", *map(str, batch_sizes), "--widths", "4", "16", "--", "sglang", "serve", "--speculative-algorithm", "DSPINE"])
    tuner.main()
    assert len(observed) == 3
    saved = json.loads(output.read_text())
    assert saved["confidence_calibration"] == calibration
    full = {str(bs): bs * 16 for bs in batch_sizes}
    snapped = {"2": 16} if batch_sizes == [2] else {"3": 16, "7": 32, "17": 64}
    actual = snapped if graph_backend == "full" else full
    assert observed[0]["mode_policies"]["sampled"]["token_budgets"] == full  # full16 runs first
    assert observed[1]["mode_policies"]["sampled"]["token_budgets"] == actual
    assert [(r["width"], r["token_budget"]) for r in saved["measurements"]] == [(16, full[str(bs)]) for bs in batch_sizes] + [(4, actual[str(bs)]) for bs in batch_sizes]
    assert saved["policy"] == {"verify_widths": {}, "token_budgets": {}}
    assert saved["mode_policies"]["greedy"] == saved["policy"]
    assert saved["mode_policies"]["sampled"]["token_budgets"] == full
    assert saved["validation"]["greedy"] == []
    assert [r["token_budget"] for r in saved["validation"]["sampled"]] == [full[str(bs)] for bs in batch_sizes]
    assert all(r["mode"] == "sampled" for r in saved["measurements"] + saved["validation"]["sampled"])
