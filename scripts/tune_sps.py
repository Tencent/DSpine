"""Measure DSpine verify budgets on the deployment GPU and representative prefixes."""

import argparse
import contextlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

# Also works directly from a source checkout, before an editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sglang_dspine.verification.calibration import load_calibration_tables
from sglang_dspine.verification.policy import SpsPolicy, hardware_identity
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sps_search import (
    aggregate, current_budget, paired_decision, prefer_budget, replace_budget,
    run_search, split_tasks as _split_tasks,
)


def option_value(command, name):
    """Read the native CLI's separated/equal forms, with last-value precedence."""
    result = None
    for index, argument in enumerate(command):
        if argument == name:
            if index + 1 == len(command) or command[index + 1].startswith("--"):
                raise ValueError(f"Missing value for {name}")
            result = command[index + 1]
        elif argument.startswith(name + "="):
            result = argument.split("=", 1)[1]
    return result


class Credentials:
    def __init__(self, command=()):
        self.api_key = option_value(command, "--api-key")
        self.admin_api_key = option_value(command, "--admin-api-key")
        # Include overridden values: a launch log may still echo the full argv.
        self.secrets = set()
        for index, argument in enumerate(command):
            if argument in ("--api-key", "--admin-api-key"):
                self.secrets.add(command[index + 1])
            elif argument.split("=", 1)[0] in ("--api-key", "--admin-api-key"):
                self.secrets.add(argument.split("=", 1)[1])
        self.secrets.discard("")
        if any("\n" in value or "\r" in value for value in self.secrets):
            raise ValueError("Authentication keys cannot contain line breaks")

    def redact(self, value):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                name = str(key).lower().replace("-", "_")
                if name in {"api_key", "admin_api_key", "authorization", "access_token", "secret_key"}:
                    result[key] = "[REDACTED]"
                elif name in {"command", "launch_command", "message", "body", "body_read_error"} and isinstance(item, str):
                    result[key] = self.redact_text(item)
                else:
                    result[key] = self.redact(item)
            return result
        if isinstance(value, (list, tuple)):
            result, secret_value = [], False
            for item in value:
                if secret_value:
                    result.append("[REDACTED]")
                    secret_value = False
                elif isinstance(item, str) and item in ("--api-key", "--admin-api-key"):
                    result.append(item)
                    secret_value = True
                elif isinstance(item, str) and item.split("=", 1)[0] in ("--api-key", "--admin-api-key"):
                    result.append(item.split("=", 1)[0] + "=[REDACTED]")
                else:
                    result.append(self.redact(item))
            return result
        return value

    def redact_text(self, value):
        # Text logs/errors can echo credentials anywhere; structured protocol
        # fields above must survive even when a key is a public word or digit.
        for secret in sorted(self.secrets, key=len, reverse=True):
            for spelling in (secret, json.dumps(secret)[1:-1], repr(secret)[1:-1]):
                value = value.replace(spelling, "[REDACTED]")
        return value

    def headers(self, url):
        # The endpoints used here are native NORMAL endpoints: api_key wins
        # when both are configured. Admin credentials never replace that key.
        key = self.api_key
        return {"Authorization": "Bearer " + key} if key else {}


_credentials = Credentials()


class AuthenticationError(RuntimeError):
    def __init__(self, status, path):
        self.status = status
        super().__init__(f"Server rejected authentication (HTTP {status}) at {path}")


def request(url, payload=None, timeout=None, *, allow_text=False, admin=False):
    data = None if payload is None else json.dumps(payload).encode()
    key = (_credentials.admin_api_key or _credentials.api_key) if admin else _credentials.api_key
    headers = {"Authorization": "Bearer " + key} if key else {}
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read()
            try:
                return json.loads(body)
            except ValueError:
                if not allow_text:
                    raise
                return {"body": body.decode() if isinstance(body, bytes) else body}
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise AuthenticationError(exc.code, urlsplit(url).path) from None
        raise


def load_prefixes(path):
    rows = [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]
    if not rows:
        raise ValueError("Requests JSONL is empty")
    for row in rows:
        ids = row.get("input_ids")
        if (
            not isinstance(ids, list)
            or not ids
            or any(type(t) is not int or t < 0 for t in ids)
        ):
            raise ValueError("Each row needs a nonempty input_ids list of token IDs")
        params = row.setdefault("sampling_params", {"temperature": 0.0})
        limit = params.get("max_new_tokens")
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError("Per-row max_new_tokens must be a positive integer")
        if any(k in params for k in ("json_schema", "regex", "ebnf", "custom_params")):
            raise ValueError("SPS profiling requires unconstrained sampling")
    return rows


def prefix_mode(row):
    # load_prefixes supplies temperature=0 only when sampling_params is absent.
    # Native SamplingParams normalizes tiny temperatures to top_k=1 before
    # validation; SamplingBatchInfo then uses top_k<=1 to identify greedy mode.
    params = row["sampling_params"]
    temperature = params.get("temperature", 1.0)
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be a non-negative finite number")
    if temperature < 1e-6:
        return "greedy"
    top_k = params.get("top_k", -1)
    if top_k != -1 and top_k < 1:
        raise ValueError("top_k must be -1 (disable) or at least 1")
    return "greedy" if top_k == 1 else "sampled"


def group_prefixes(prefixes):
    groups = {"greedy": [], "sampled": []}
    for row in prefixes:
        groups[prefix_mode(row)].append(row)
    return groups


def select_policy(measurements, min_gain=0.03):
    budgets = {}
    for bs in sorted({row["batch_size"] for row in measurements}):
        rows = [row for row in measurements if row["batch_size"] == bs]
        decision = prefer_budget(rows, bs * 16, min_gain)
        if decision["reason"] != "insufficient_baseline":
            budgets[bs] = decision["selected_budget"]
    return SpsPolicy(token_budgets=budgets)


def native_token_grid(info):
    """The native decode runner captures bs * draft-block tokens per tier."""
    block = info.get("speculative_num_draft_tokens")
    decode = (info.get("cuda_graph_config") or {}).get("decode") or {}
    if (
        info.get("disable_cuda_graph") or info.get("disable_decode_cuda_graph")
        or decode.get("backend") in (None, "disabled")
        or type(block) is not int or block < 1
    ):
        return []
    return sorted({bs * block for bs in (decode.get("bs") or []) if type(bs) is int and bs > 0})


def candidate_budgets(batch_sizes, width, token_grid, block=16):
    """Snap candidates to exact captured tiers, retaining the full-block baseline."""
    budgets = {}
    request_slots = max(token_grid, default=0) // block
    for bs in batch_sizes:
        legal = [tier for tier in token_grid if bs <= tier <= bs * block] if bs <= request_slots else []
        budgets[bs] = (
            min(legal, key=lambda tier: (abs(tier - width * bs), -tier))
            if width != block and legal else bs * block
        )
    return budgets


@contextlib.contextmanager
def server(command, profile, folder, port, startup_timeout):
    global _credentials
    # Only this script's process group is stopped; no existing service is reused.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))
    folder.mkdir(parents=True, exist_ok=True)
    argv = command + [
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--speculative-dspark-sps-table-path",
        str(profile),
    ]
    credentials = Credentials(command)
    (folder / "command.json").write_text(json.dumps(credentials.redact(argv), indent=2) + "\n")
    with (folder / "server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True, text=True, encoding="utf-8", errors="replace",
        )

        def drain_log():
            # Never put the unsanitized native startup/argv log on disk.
            for line in process.stdout:
                log.write(credentials.redact_text(line))
                log.flush()

        reader = threading.Thread(target=drain_log, daemon=True)
        reader.start()
        previous, _credentials = _credentials, credentials
        try:
            deadline = time.monotonic() + startup_timeout
            while True:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Server exited ({process.returncode}); inspect {folder / 'server.log'}"
                    )
                try:
                    request(f"http://127.0.0.1:{port}/health", timeout=2, allow_text=True)
                    break
                except (OSError, ValueError, urllib.error.URLError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"Server startup timed out; inspect {folder / 'server.log'}"
                        )
                    time.sleep(1)
            yield f"http://127.0.0.1:{port}"
        finally:
            # The parent can exit before its scheduler children.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            # Reap any remaining members of this newly-created group.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            reader.join()
            process.stdout.close()
            _credentials = previous


def queue_payloads(prefixes, bs, repeat, count, maximum):
    """Method-independent seeds remain paired across budget candidates."""
    mode = prefix_mode(prefixes[0])
    jobs = []
    for index in range(count):
        source, copy = index % len(prefixes), index // len(prefixes)
        row = prefixes[source]
        identity = ["sps-continuous-v1", mode, bs, repeat, source, copy,
                    row["input_ids"], row["sampling_params"]]
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        seed = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little") & ((1 << 63) - 1)
        payload = {"rid": f"sps-{mode}-b{bs}-r{repeat}-i{index}", "input_ids": row["input_ids"],
                   "sampling_params": {**row["sampling_params"], "sampling_seed": seed,
                                       "max_new_tokens": min(maximum, row["sampling_params"].get("max_new_tokens") or maximum)},
                   "stream": False, "return_spec_tokens_details": True}
        jobs.append({"source_index": source, "copy": copy, "payload": payload})
    if len({job["payload"]["sampling_params"]["sampling_seed"] for job in jobs}) != count:
        raise RuntimeError("Sampling seed collision; no requests were sent")
    return jobs


def checked_counts(response, prompt_tokens, rid, maximum):
    errors = []
    if not isinstance(response, dict):
        return {}, ["Expected one native response object"]
    meta, ids = response.get("meta_info"), response.get("output_ids")
    if not isinstance(meta, dict) or not isinstance(ids, list) or not ids:
        return {}, ["Missing native metadata or output tokens"]
    finish = meta.get("finish_reason") or {}
    if response.get("error") or not isinstance(finish, dict) or finish.get("type") not in ("stop", "length"):
        errors.append("Generation failed or did not finish normally")
    if meta.get("prompt_tokens") != prompt_tokens or meta.get("completion_tokens") != len(ids):
        errors.append("Native prompt/completion count mismatch")
    if meta.get("id") != rid:
        errors.append("Native response ID does not match request rid")
    if len(ids) > maximum:
        errors.append("Native response exceeds requested output cap")
    prefill_eos = len(ids) == 1 and isinstance(finish, dict) and finish.get("type") == "stop"
    counts = {}
    for name in ("spec_num_correct_drafts", "spec_num_proposed_drafts", "spec_verify_ct"):
        value = meta.get(name)
        if value is None and prefill_eos:
            value = 0  # Native prefill can emit EOS before its first draft round.
        if type(value) is not int or value < 0:
            errors.append("Missing or invalid " + name)
        else:
            counts[name] = value
    if len(counts) == 3:
        accepted, proposed, verifies = (counts[k] for k in ("spec_num_correct_drafts", "spec_num_proposed_drafts", "spec_verify_ct"))
        if accepted > proposed or (verifies == 0 and (not prefill_eos or accepted or proposed)):
            errors.append("Inconsistent accepted/proposed/verify counters")
        counts["prefill_only_eos"] = prefill_eos and verifies == 0
    return counts, errors


def run_queue(url, jobs, bs, mode, repeat, raw, request_timeout=None):
    barrier = threading.Barrier(min(bs, len(jobs)))
    lock, tracker = threading.Lock(), {"entered": 0, "active": 0, "peak": 0}
    auth_failed = threading.Event()

    def generate(job):
        payload = job["payload"]
        started, begin = time.time(), time.perf_counter()
        response = error = None
        try:
            with lock:
                tracker["entered"] += 1
                first = tracker["entered"] <= bs
            if first:
                barrier.wait(timeout=60)
            with lock:
                tracker["active"] += 1
                tracker["peak"] = max(tracker["peak"], tracker["active"])
            try:
                if auth_failed.is_set():
                    raise RuntimeError("Queue stopped after an authentication failure")
                response = request(url + "/generate", payload, timeout=request_timeout)
            finally:
                with lock:
                    tracker["active"] -= 1
        except Exception as exc:
            if isinstance(exc, AuthenticationError):
                auth_failed.set()
            error = {"type": type(exc).__name__, "message": str(exc)}
            if isinstance(exc, AuthenticationError):
                error["status"] = exc.status
            if isinstance(exc, urllib.error.HTTPError):
                error["status"] = exc.code
                try:
                    error["body"] = exc.read().decode(errors="replace")
                except Exception as body_error:
                    error["body_read_error"] = str(body_error)
        counts, errors = checked_counts(response, len(payload["input_ids"]), payload["rid"], payload["sampling_params"]["max_new_tokens"]) if error is None else ({}, [])
        return {"mode": mode, "batch_size": bs, "repeat": repeat, "excluded_from_metrics": repeat < 0,
                "source_index": job["source_index"], "copy": job["copy"], "rid": payload["rid"],
                "input_ids_sha256": hashlib.sha256(json.dumps(payload["input_ids"]).encode()).hexdigest(),
                "sampling_params": payload["sampling_params"], "started_unix_s": started,
                "finished_unix_s": time.time(), "request_wall_seconds": time.perf_counter() - begin,
                "http_status": error.get("status") if error else 200,
                "response": response, "http_error": error, "validation_errors": errors, "counts": counts}

    records = []
    started, begin = time.time(), time.perf_counter()
    with ThreadPoolExecutor(max_workers=bs) as pool:
        futures = [pool.submit(generate, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            records.append(row)
            raw.write(json.dumps(_credentials.redact(row)) + "\n")
            raw.flush()  # Retain every HTTP/counter error before rejecting the queue.
    wall = time.perf_counter() - begin
    valid = [r for r in records if not r["http_error"] and not r["validation_errors"]]
    lengths = [len(r["response"]["output_ids"]) for r in valid]
    tokens = sum(lengths)
    accepted = sum(r["counts"]["spec_num_correct_drafts"] for r in valid)
    proposed = sum(r["counts"]["spec_num_proposed_drafts"] for r in valid)
    verifies = sum(r["counts"]["spec_verify_ct"] for r in valid)
    return {"mode": mode, "batch_size": bs, "repeat": repeat, "excluded_from_metrics": repeat < 0,
            "status": "PASS" if len(valid) == len(jobs) else "FAIL", "requests": len(records),
            "error_requests": len(records) - len(valid), "peak_concurrent_http": tracker["peak"],
            "queue_started_unix_s": started, "wall_seconds": wall, "output_tokens": tokens,
            "tokens_per_second": tokens / wall, "accepted_drafts": accepted, "proposed_drafts": proposed,
            "verify_calls": verifies, "prefill_only_eos_requests": sum(r["counts"].get("prefill_only_eos", False) for r in valid),
            "acceptance_length": accepted / verifies + 1 if verifies else None,
            "emitted_tokens_per_verify": tokens / verifies if verifies else None,
            "output_lengths": {"min": min(lengths, default=0), "max": max(lengths, default=0),
                               "median": statistics.median(lengths) if lengths else None},
            "hit_max_new_tokens_requests": sum(len(r["response"]["output_ids"]) == r["sampling_params"]["max_new_tokens"] for r in valid),
            "finish_counts": dict(Counter(r["response"]["meta_info"]["finish_reason"]["type"] for r in valid))}


def measure(url, prefixes, batch_sizes, repeats, max_new_tokens, raw_path, request_timeout=None):
    modes = {prefix_mode(row) for row in prefixes}
    if len(modes) != 1:
        raise ValueError("Measure one nonempty greedy or sampled request family at a time")
    mode = modes.pop()
    info = request(url + "/server_info")
    if info.get("speculative_algorithm") != "DSPINE":
        raise ValueError("SPS tuning requires the actual DSPINE serving path")
    for state in info.get("internal_states", []):
        limit = state.get("effective_max_running_requests_per_dp")
        if limit and limit < max(batch_sizes):
            raise ValueError("Server running-request capacity is below the requested batch size")
    result, summaries = [], []
    raw_path = Path(raw_path)
    with raw_path.open("w") as raw:
        for bs in batch_sizes:
            rounds = []
            for repeat in range(-1, repeats):
                # Independent requests keep replacing completed ones. A single
                # exact-bs wave otherwise spends nearly all time on its tail.
                count = bs if repeat < 0 else max(len(prefixes), 4 * bs)
                maximum = min(max_new_tokens, 32) if repeat < 0 else max_new_tokens
                jobs = queue_payloads(prefixes, bs, repeat, count, maximum)
                summary = run_queue(url, jobs, bs, mode, repeat, raw, request_timeout)
                summary["jobs_sha256"] = hashlib.sha256(json.dumps(
                    [job["payload"] for job in jobs], sort_keys=True,
                    separators=(",", ":"),
                ).encode()).hexdigest()
                summaries.append(summary)
                raw_path.with_suffix(".summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
                if summary["status"] != "PASS":
                    raise RuntimeError(f"HTTP/counter errors retained at {raw_path}; refusing this profile")
                if repeat >= 0:
                    rounds.append(summary)
            wall = sum(r["wall_seconds"] for r in rounds)
            result.append({"mode": mode, "batch_size": bs,
                "tokens_per_second": sum(r["output_tokens"] for r in rounds) / wall,
                "speculative_verification_observed": any(r["verify_calls"] for r in rounds), "rounds": rounds})
    return result


def validate_combined_policy(baseline, candidate):
    """Gate a complete table; nominal-B trials also execute smaller actual B."""
    comparisons = []
    for mode in baseline:
        before = {row["batch_size"]: row for row in baseline[mode]}
        after = {row["batch_size"]: row for row in candidate[mode]}
        if len(before) != len(baseline[mode]) or len(after) != len(candidate[mode]) or before.keys() != after.keys():
            raise ValueError("Final validation must cover the complete baseline matrix")
        for bs, reference in before.items():
            trial = after[bs]
            for rows in (reference, trial):
                rounds = rows.get("rounds", [])
                if not rounds or any(r.get("status") != "PASS" or not r.get("jobs_sha256") for r in rounds):
                    raise ValueError("Final validation needs successful paired request receipts")
            signature = lambda row: [(r["repeat"], r["requests"], r["jobs_sha256"]) for r in row["rounds"]]
            if signature(reference) != signature(trial):
                raise ValueError("Final validation requests, seeds or repeats differ from full16")
            old, new = reference["tokens_per_second"], trial["tokens_per_second"]
            if not all(math.isfinite(rate) and rate > 0 for rate in (old, new)):
                raise ValueError("Final validation throughput must be positive and finite")
            ratio = new / old
            comparisons.append({"mode": mode, "batch_size": bs,
                "full16_tokens_per_second": old, "combined_tokens_per_second": new,
                "throughput_ratio": ratio, "material_regression": ratio < 0.97})
    fallback = any(row["material_regression"] for row in comparisons)
    return {"selected_profile": "full16" if fallback else "combined",
        "reason": "combined_policy_regressed" if fallback else "combined_policy_within_tolerance",
        "regression_tolerance": 0.03, "comparisons": comparisons}



def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def split_tasks(prefixes, holdout=None):
    mark = lambda rows: [{**row, "_mode": prefix_mode(row)} for row in rows]
    return _split_tasks(mark(prefixes), None if holdout is None else mark(holdout))


def _native_batch_evidence(path, start, wall, bs):
    import re
    observed = []
    for line in Path(path).read_text(errors="replace").splitlines():
        stamp = re.match(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", line)
        match = re.search(r"Decode batch, #running-req: (\d+)", line)
        if stamp and match:
            epoch = time.mktime(time.strptime(stamp[1], "%Y-%m-%d %H:%M:%S"))
            if start <= epoch and epoch + 1 <= start + wall:
                observed.append(int(match[1]))
    return {"source": "native periodic decode log, not a per-round trace",
            "histogram": dict(Counter(observed)), "samples": len(observed),
            "observed_target_batch": bs in observed, "target_batch": bs}


def run_profile_trials(command, document, prefixes, bs, repeat, count, maximum, folder,
                       port, startup_timeout, request_timeout=None):
    """One complete paired pass; an experiment runner may supply another service factory."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    profile = folder / "sps.json"
    credentials = Credentials(command)
    save_json(profile, credentials.redact(document))
    mode = prefix_mode(prefixes[0])
    with server(command, profile, folder, port, startup_timeout) as url:
        info = request(url + "/server_info", timeout=60)
        if info.get("speculative_algorithm") != "DSPINE" or info.get("speculative_num_draft_tokens") != 16:
            raise ValueError("Expected the native DSPINE block16 service")
        limits = [x.get("effective_max_running_requests_per_dp") for x in info.get("internal_states", [])]
        if limits and any(type(x) is not int or x < bs for x in limits):
            raise ValueError("Native request capacity is below the requested actual batch")
        save_json(folder / "server_info.json", credentials.redact(info))
        with (folder / "requests.jsonl").open("w") as raw:
            request(url + "/flush_cache", {}, timeout=60, allow_text=True, admin=True)
            warm = queue_payloads(prefixes, bs, -1, max(16, bs), min(64, maximum))
            warmup = run_queue(url, warm, bs, mode, -1, raw, request_timeout)
            save_json(folder / "warmup.json", warmup)
            if warmup["status"] != "PASS":
                raise RuntimeError(f"Warmup errors retained in {folder}")
            request(url + "/flush_cache", {}, timeout=60, allow_text=True, admin=True)
            jobs = queue_payloads(prefixes, bs, repeat, count, maximum)
            result = run_queue(url, jobs, bs, mode, repeat, raw, request_timeout)
        if result["status"] != "PASS":
            save_json(folder / "SUMMARY.json", result)
            raise RuntimeError(f"Native HTTP/counter errors retained in {folder}")
        evidence = _native_batch_evidence(folder / "server.log", result["queue_started_unix_s"], result["wall_seconds"], bs)
    result.update(jobs_sha256=hashlib.sha256(json.dumps([j["payload"] for j in jobs], sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                  profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(), actual_batch_evidence=evidence)
    if not evidence["observed_target_batch"] or result["peak_concurrent_http"] != bs:
        result["status"] = "INSUFFICIENT_ACTUAL_BATCH"
    save_json(folder / "SUMMARY.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True, help="Representative tokenized task prefixes with sampling_params")
    parser.add_argument("--holdout-requests", type=Path, help="Independent task prefixes; otherwise reserve one third of tasks")
    parser.add_argument("--baseline-profile", type=Path, help="Current deployment SPS profile; default verifies full blocks")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence-calibration", type=Path)
    parser.add_argument("--batch-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--widths", type=int, nargs="+", default=[4, 8, 12, 16])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--min-seconds", type=float, default=30)
    parser.add_argument("--min-requests", type=int, default=128)
    parser.add_argument("--min-gain", type=float, default=0.03)
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--request-timeout", type=float, default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- followed by the deployment command")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        credentials = Credentials(command)
        if not command or option_value(command, "--speculative-algorithm") != "DSPINE":
            raise ValueError("Supply a deployment command with --speculative-algorithm DSPINE")
        if any(a.split("=", 1)[0] in ("--host", "--port", "--speculative-dspark-sps-table-path") for a in command):
            raise ValueError("The tuner supplies host, port and SPS-table options")
        args.batch_sizes, args.widths = sorted(set(args.batch_sizes)), sorted(set(args.widths))
        if not args.batch_sizes or min(args.batch_sizes) < 1 or not set(args.widths) <= set(range(1,17)) or 16 not in args.widths:
            raise ValueError("Use positive batch sizes and candidate widths1–16 including16")
        if args.repeats < 2 or min(args.max_new_tokens, args.min_requests) < 1:
            raise ValueError("Use at least two repeats and positive request/token limits")
        if any(not math.isfinite(x) or x <= 0 for x in (args.min_seconds,args.startup_timeout,args.min_gain)) or args.min_gain >= 1:
            raise ValueError("Duration/timeouts must be positive and min-gain must be between0 and1")
        if args.request_timeout is not None and (not math.isfinite(args.request_timeout) or args.request_timeout <= 0):
            raise ValueError("Request timeout must be positive and finite")
        prefixes = load_prefixes(args.requests)
        held = load_prefixes(args.holdout_requests) if args.holdout_requests else None
        tune, holdout, partition = split_tasks(prefixes, held)
    except ValueError as error:
        parser.error(str(error))
    hardware = hardware_identity(0)
    base = {"format":"dspine-sps-v2", "hardware":hardware, "policy":SpsPolicy().to_dict(), "mode_policies":{}}
    if args.baseline_profile:
        default, modes = SpsPolicy.load_with_modes(args.baseline_profile, hardware)
        old = json.loads(args.baseline_profile.read_text())
        base.update(policy=default.to_dict(),mode_policies={mode:p.to_dict() for mode,p in modes.items()})
        if "confidence_calibration" in old:
            base["confidence_calibration"] = old["confidence_calibration"]
    if args.confidence_calibration:
        calibration = json.loads(args.confidence_calibration.read_text())
        load_calibration_tables(calibration)
        base["confidence_calibration"] = calibration
    output = args.output.resolve()
    folder = output.with_name(output.stem + ".runs")
    folder.mkdir(parents=True, exist_ok=False)
    save_json(folder / "INPUTS.json", {**partition, "requests_sha256":hashlib.sha256(args.requests.read_bytes()).hexdigest(),
        "holdout_sha256":hashlib.sha256(args.holdout_requests.read_bytes()).hexdigest() if args.holdout_requests else None,
        "max_new_tokens":args.max_new_tokens,
        "effective_output_caps":{name:dict(Counter(min(args.max_new_tokens, row["sampling_params"].get("max_new_tokens") or args.max_new_tokens) for row in rows))
                                 for name,rows in (("tune",tune),("holdout",holdout))}})
    probe = folder / "probe.json"
    save_json(probe, base)
    with server(command, probe, folder / "probe", args.port, args.startup_timeout) as url:
        info = request(url + "/server_info", timeout=60)
        args.token_grid = native_token_grid(info)
        save_json(folder / "server_info.json", credentials.redact(info))
    def trial(document, rows, bs, repeat, count, destination):
        return run_profile_trials(command, document, rows, bs, repeat, count, args.max_new_tokens,
                                  destination, args.port, args.startup_timeout, args.request_timeout)
    try:
        selected, report = run_search(args, base, tune, holdout, folder, trial, candidate_budgets, save_json)
    except Exception as error:
        save_json(folder / "ERROR.json", {"status":"FAIL", "error":credentials.redact_text(repr(error))})
        raise
    selected["tuning"] = {"status":report["status"], "selection_status":report["selection_status"],
                           "report":str(folder / "RESULTS.json"), "min_gain":args.min_gain}
    save_json(output, credentials.redact(selected))
    print(json.dumps({"profile":str(output), "status":report["status"], "selection_status":report["selection_status"]}),flush=True)


if __name__ == "__main__":
    main()
