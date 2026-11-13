"""Keep the two sampler families independent throughout SPS tuning and replay."""

import contextlib
import importlib.util
import io
import json
import sys
import threading
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


@pytest.fixture
def tuner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "tune_sps.py"
    spec = importlib.util.spec_from_file_location("tune_sps_modes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_request_family_defaults_follow_serving_temperature(tuner, tmp_path):
    path = tmp_path / "prefixes.jsonl"
    rows = [
        {"input_ids": [1]},
        {"input_ids": [2], "sampling_params": {}},
        {"input_ids": [3], "sampling_params": {"temperature": 0}},
        {"input_ids": [4], "sampling_params": {"temperature": 0.7}},
    ]
    path.write_text("\n".join(map(json.dumps, rows)))
    loaded = tuner.load_prefixes(path)
    assert loaded[0]["sampling_params"] == {"temperature": 0.0}
    groups = tuner.group_prefixes(loaded)
    assert [row["input_ids"] for row in groups["greedy"]] == [[1], [3]]
    assert [row["input_ids"] for row in groups["sampled"]] == [[2], [4]]


@pytest.mark.parametrize("params,expected", [
    ({"temperature": 0.7, "top_k": 1}, "greedy"),
    ({"top_k": 1}, "greedy"),
    ({"temperature": 1e-7, "top_k": 5}, "greedy"),
    ({"temperature": 1e-6, "top_k": 5}, "sampled"),
    ({"temperature": 0.7, "top_k": -1}, "sampled"),
    # Native normalization overrides this top_k before its validity check.
    ({"temperature": 0.0, "top_k": 0}, "greedy"),
])
def test_prefix_mode_matches_native_greedy_normalization(tuner, params, expected):
    assert tuner.prefix_mode({"sampling_params": params}) == expected


@pytest.mark.parametrize("params", [
    {"temperature": 0.7, "top_k": 0},
    {"temperature": 0.7, "top_k": -2},
    {"temperature": -1, "top_k": 1},
])
def test_prefix_mode_rejects_native_invalid_settings(tuner, params):
    with pytest.raises(ValueError):
        tuner.prefix_mode({"sampling_params": params})


@pytest.mark.parametrize("temperature", [0.0, 0.7])
def test_measure_cycles_a_small_family_into_continuous_single_requests(
    tuner, monkeypatch, tmp_path, temperature
):
    mode = "greedy" if temperature == 0 else "sampled"
    prefixes = [{"input_ids": [42], "sampling_params": {"temperature": temperature}}]
    payloads = []

    def request(url, payload=None, **kwargs):
        if url.endswith("/server_info"):
            return {"speculative_algorithm": "DSPINE"}
        assert url.endswith("/generate")
        payloads.append(payload)
        return native_response(payload["rid"])

    monkeypatch.setattr(tuner, "request", request)
    path = tmp_path / "requests.jsonl"
    measured = tuner.measure("http://cpu", prefixes, [3, 7], 1, 2, path)
    assert len(payloads) == 3 + 12 + 7 + 28
    assert all(payload["input_ids"] == [42] for payload in payloads)
    assert all(payload["sampling_params"]["temperature"] == temperature for payload in payloads)
    assert len({payload["sampling_params"]["sampling_seed"] for payload in payloads}) == len(payloads)
    assert [row["mode"] for row in measured] == [mode, mode]
    assert [row["rounds"][0]["output_tokens"] for row in measured] == [24, 56]
    assert all(row["tokens_per_second"] > 0 for row in measured)
    raw = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["mode"] for row in raw] == [mode] * len(payloads)
    assert sum(row["excluded_from_metrics"] for row in raw) == 10
    assert all(not row["validation_errors"] and row["http_status"] == 200 for row in raw)


def native_response(rid):
    return {"output_ids": [11, 12], "meta_info": {
        "id": rid, "prompt_tokens": 1, "completion_tokens": 2, "spec_num_correct_drafts": 1,
        "spec_num_proposed_drafts": 15, "spec_verify_ct": 1, "finish_reason": {"type": "stop"},
    }}


@pytest.mark.parametrize("response_id,maximum,message", [
    ("wrong-request", 2, "request rid"),
    ("expected-request", 1, "output cap"),
])
def test_native_response_cannot_cross_requests_or_exceed_the_cap(tuner, response_id, maximum, message):
    _, errors = tuner.checked_counts(native_response(response_id), 1, "expected-request", maximum)
    assert any(message in error for error in errors)


def test_queue_seeds_pair_across_budgets_and_change_for_every_copy_repeat(tuner):
    prefixes = [{"input_ids": [42], "sampling_params": {"temperature": 1, "sampling_seed": 9}}]
    first = tuner.queue_payloads(prefixes, 3, 0, 12, 2048)
    again = tuner.queue_payloads(prefixes, 3, 0, 12, 2048)
    second = tuner.queue_payloads(prefixes, 3, 1, 12, 2048)
    seeds = lambda rows: [r["payload"]["sampling_params"]["sampling_seed"] for r in rows]
    assert seeds(first) == seeds(again)
    assert len(set(seeds(first))) == 12
    assert not (set(seeds(first)) & set(seeds(second)))
    assert prefixes[0]["sampling_params"]["sampling_seed"] == 9


def test_continuous_queue_replaces_one_finished_request_before_other_wave_members(tuner, monkeypatch, tmp_path):
    ready = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    refilled, lock, calls = threading.Event(), threading.Lock(), []

    def request(url, payload=None, **kwargs):
        if url.endswith("/server_info"):
            return {"speculative_algorithm": "DSPINE"}
        if payload["sampling_params"]["max_new_tokens"] == 64:
            with lock:
                index = len(calls)
                calls.append(payload)
            if index < 2:
                ready[index].set()
                assert release[index].wait(3)
            elif index == 2:
                refilled.set()
                release[1].set()
        return native_response(payload["rid"])

    monkeypatch.setattr(tuner, "request", request)
    prefixes = [{"input_ids": [42], "sampling_params": {"temperature": 1}}]
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(tuner.measure, "http://cpu", prefixes, [2], 2, 64, tmp_path / "raw.jsonl")
        try:
            assert all(event.wait(2) for event in ready)
            release[0].set()
            assert refilled.wait(2)  # Second initial request remains blocked until refill.
            result = future.result(timeout=5)
        finally:
            for event in release:
                event.set()
    rounds = result[0]["rounds"]
    assert len(calls) == 16 and all(r["requests"] == 8 for r in rounds)
    assert rounds[0]["peak_concurrent_http"] == 2
    assert result[0]["tokens_per_second"] == sum(r["output_tokens"] for r in rounds) / sum(r["wall_seconds"] for r in rounds)


def test_prefill_only_eos_is_valid_but_cannot_select_a_dynamic_budget(tuner, monkeypatch, tmp_path):
    def request(url, payload=None, **kwargs):
        if url.endswith("/server_info"):
            return {"speculative_algorithm": "DSPINE"}
        return {"output_ids": [2], "meta_info": {"id": payload["rid"], "prompt_tokens": 1, "completion_tokens": 1,
                "finish_reason": {"type": "stop", "matched": 2}}}

    monkeypatch.setattr(tuner, "request", request)
    result = tuner.measure("http://cpu", [{"input_ids": [42], "sampling_params": {"temperature": 0}}], [3], 1, 64, tmp_path / "raw.jsonl")
    assert result[0]["rounds"][0]["prefill_only_eos_requests"] == 12
    assert result[0]["rounds"][0]["verify_calls"] == 0
    assert result[0]["speculative_verification_observed"] is False
    assert tuner.select_policy([{**result[0], "token_budget": 32}]).budget(3) == 48


def test_http_and_counter_errors_are_all_saved_before_queue_fails(tuner, monkeypatch, tmp_path):
    def request(url, payload=None, **kwargs):
        if url.endswith("/server_info"):
            return {"speculative_algorithm": "DSPINE"}
        response = native_response(payload["rid"])
        if payload["sampling_params"]["max_new_tokens"] > 32:
            if payload["input_ids"] == [42]:
                raise urllib.error.HTTPError(url, 503, "test error", {}, io.BytesIO(b"retained body"))
            response["meta_info"]["spec_num_correct_drafts"] = 16
        return response

    monkeypatch.setattr(tuner, "request", request)
    path = tmp_path / "raw.jsonl"
    prefixes = [{"input_ids": [x], "sampling_params": {"temperature": 1}} for x in (42, 43)]
    with pytest.raises(RuntimeError, match="errors retained"):
        tuner.measure("http://cpu", prefixes, [2], 1, 64, path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 10  # Two excluded warmup responses and every measured request.
    assert sum(bool(r["http_error"]) for r in rows) == 4
    assert sum(bool(r["validation_errors"]) for r in rows) == 4
    assert {r["http_error"]["body"] for r in rows if r["http_error"]} == {"retained body"}
    summary = json.loads(path.with_suffix(".summary.json").read_text())[-1]
    assert summary["status"] == "FAIL" and summary["requests"] == summary["error_requests"] == 8


def test_measure_rejects_mixed_family_batches(tuner, monkeypatch, tmp_path):
    monkeypatch.setattr(tuner, "request", lambda *args: pytest.fail("Must reject before serving"))
    prefixes = [
        {"input_ids": [1], "sampling_params": {"temperature": temperature}}
        for temperature in (0, 0.7)
    ]
    with pytest.raises(ValueError, match="one nonempty"):
        tuner.measure("http://cpu", prefixes, [2], 1, 2, tmp_path / "requests.jsonl")


def test_selection_requires_repeated_paired_full_block_reference(tuner):
    def row(budget, rate):
        return {"batch_size": 7, "token_budget": budget, "tokens_per_second": rate,
            "status": "PASS", "speculative_verification_observed": True,
            "rounds": [{"repeat": repeat, "requests": 128, "jobs_sha256": str(repeat),
                        "tokens_per_second": rate} for repeat in range(2)]}
    policy = tuner.select_policy([row(32, 200), row(112, 100)])
    assert policy.budget(7) == 32
    assert policy.budget(3) == 48
    assert tuner.select_policy([row(32, 200)]).budget(7) == 112
    assert tuner.select_policy([]).to_dict() == {"verify_widths": {}, "token_budgets": {}}


def test_explicit_row_cap_limits_only_that_request(tuner):
    rows = [{"input_ids": [1], "sampling_params": {"temperature": 0.37, "max_new_tokens": 512}},
            {"input_ids": [2], "sampling_params": {"temperature": 0.37}}]
    assert [j["payload"]["sampling_params"]["max_new_tokens"]
            for j in tuner.queue_payloads(rows, 2, 0, 2, 4096)] == [512, 4096]
    assert [j["payload"]["sampling_params"]["max_new_tokens"]
            for j in tuner.queue_payloads(rows, 2, 0, 2, 64)] == [64, 64]
    assert rows[0]["sampling_params"]["max_new_tokens"] == 512


def test_turnkey_search_requires_independent_tasks_before_starting_service(tuner, monkeypatch, tmp_path):
    requests = tmp_path / "one-task.jsonl"
    requests.write_text(json.dumps({"task_id": "a", "input_ids": [1]}) + "\n")
    monkeypatch.setattr(tuner, "hardware_identity", lambda device: pytest.fail("Must partition before GPU access"))
    monkeypatch.setattr(sys, "argv", [str(tuner.__file__), "--requests", str(requests),
        "--output", str(tmp_path / "profile.json"), "--batch-sizes", "8", "--",
        "sglang", "serve", "--speculative-algorithm", "DSPINE"])
    with pytest.raises(SystemExit) as error:
        tuner.main()
    assert error.value.code == 2
