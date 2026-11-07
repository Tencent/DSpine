"""Measure DSpine verify budgets on the deployment GPU and representative prefixes."""

import argparse
import contextlib
import hashlib
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Also works directly from a source checkout, before an editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sglang_dspine.verification.policy import SpsPolicy, hardware_identity


def request(url, payload=None, timeout=600):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


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
        if any(k in params for k in ("json_schema", "regex", "ebnf", "custom_params")):
            raise ValueError("SPS profiling requires unconstrained sampling")
    greedy = {r["sampling_params"].get("temperature", 1.0) == 0 for r in rows}
    if len(greedy) != 1:
        raise ValueError("Profile greedy and temperature sampling in separate runs")
    return rows


def select_policy(measurements):
    widths = [16] * 7
    tokens = 128
    for bs in sorted({r["batch_size"] for r in measurements}):
        eligible = [r for r in measurements if r["batch_size"] == bs]
        best = max(eligible, key=lambda r: (r["tokens_per_second"], r["width"]))
        if bs == 8:
            tokens = best["width"] * 8
        else:
            widths[bs - 1] = best["width"]
    return SpsPolicy(tuple(widths), tokens)


@contextlib.contextmanager
def server(command, profile, folder, port, startup_timeout):
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
    (folder / "command.json").write_text(json.dumps(argv, indent=2) + "\n")
    with (folder / "server.log").open("w") as log:
        process = subprocess.Popen(
            argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            deadline = time.monotonic() + startup_timeout
            while True:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Server exited ({process.returncode}); inspect {folder / 'server.log'}"
                    )
                try:
                    request(f"http://127.0.0.1:{port}/model_info", timeout=2)
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


def measure(url, prefixes, batch_sizes, repeats, max_new_tokens, raw_path):
    info = request(url + "/server_info")
    if info.get("speculative_algorithm") != "DSPINE":
        raise ValueError("SPS tuning requires the actual DSPINE serving path")
    for state in info.get("internal_states", []):
        limit = state.get("effective_max_running_requests_per_dp")
        if limit and limit < max(batch_sizes):
            raise ValueError(
                "Server running-request capacity is below the requested batch size"
            )
    result = []
    with raw_path.open("w") as raw:
        for bs in batch_sizes:
            rounds = []
            for repeat in range(
                -1, repeats
            ):  # First pass warms every prefix and sampler shape.
                tokens = 0
                wall = 0.0
                accepted = verifies = 0
                for start in range(0, len(prefixes), bs):
                    cohort = [prefixes[(start + i) % len(prefixes)] for i in range(bs)]
                    params = [
                        {**r["sampling_params"], "max_new_tokens": max_new_tokens}
                        for r in cohort
                    ]
                    payload = {
                        "input_ids": [r["input_ids"] for r in cohort],
                        "sampling_params": params,
                        "stream": False,
                        "return_spec_tokens_details": True,
                    }
                    begin = time.perf_counter()
                    rows = request(url + "/generate", payload)
                    elapsed = time.perf_counter() - begin
                    if not isinstance(rows, list):
                        rows = [rows]
                    if len(rows) != bs:
                        raise RuntimeError("Incomplete native batch response")
                    for row in rows:
                        if row.get("error") or not row.get("output_ids"):
                            raise RuntimeError(f"Generation failed: {row}")
                        meta = row["meta_info"]
                        if meta.get("finish_reason", {}).get("type") == "abort":
                            raise RuntimeError(f"Generation aborted: {meta}")
                        tokens += len(row["output_ids"])
                        accepted += meta["spec_num_correct_drafts"]
                        verifies += meta["spec_verify_ct"]
                    wall += elapsed
                    raw.write(
                        json.dumps(
                            {
                                "batch_size": bs,
                                "repeat": repeat,
                                "elapsed_seconds": elapsed,
                                "results": rows,
                            }
                        )
                        + "\n"
                    )
                    raw.flush()
                if repeat >= 0:
                    if verifies <= 0:
                        raise RuntimeError("No speculative verification observed")
                    rounds.append(
                        {
                            "output_tokens": tokens,
                            "wall_seconds": wall,
                            "tokens_per_second": tokens / wall,
                            "accepted_drafts": accepted,
                            "verify_calls": verifies,
                            "acceptance_length": accepted / verifies + 1,
                        }
                    )
            result.append(
                {
                    "batch_size": bs,
                    "tokens_per_second": statistics.median(
                        r["tokens_per_second"] for r in rounds
                    ),
                    "rounds": rounds,
                }
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requests",
        type=Path,
        required=True,
        help="JSONL of frozen input_ids and sampling_params",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--widths", type=int, nargs="+", default=[4, 8, 12, 16])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="-- followed by the ordinary sglang serve command",
    )
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or "DSPINE" not in command:
        parser.error(
            "Supply the deployment command after --, including --speculative-algorithm DSPINE"
        )
    if any(
        a.split("=", 1)[0]
        in ("--host", "--port", "--speculative-dspark-sps-table-path")
        for a in command
    ):
        parser.error(
            "Remove host, port and SPS-table options from the deployment command; the tuner supplies them"
        )
    sizes, widths = sorted(set(args.batch_sizes)), sorted(set(args.widths))
    if (
        not sizes
        or not set(sizes) <= set(range(1, 9))
        or not widths
        or not set(widths) <= set(range(2, 17, 2))
    ):
        parser.error("Use batch sizes 1–8 and even widths 2–16")
    if args.repeats < 1 or args.max_new_tokens < 1:
        parser.error("Repeats and max-new-tokens must be positive")
    if 16 not in widths:
        parser.error("Include width16 as the verify-all baseline")
    prefixes = load_prefixes(args.requests)
    if len(prefixes) < max(sizes):
        parser.error("Provide at least as many distinct prefixes as the largest batch")
    hardware = hardware_identity(0)  # Set CUDA_VISIBLE_DEVICES to the deployment GPU.
    output = args.output.resolve()
    folder = output.with_name(output.stem + ".runs")
    folder.mkdir(parents=True, exist_ok=False)
    base = {
        "format": "dspine-sps-v1",
        "hardware": hardware,
        "requests_sha256": hashlib.sha256(args.requests.read_bytes()).hexdigest(),
        "command": command,
        "max_new_tokens": args.max_new_tokens,
        "objective": "median warm-prefix complete-request tokens/second",
    }
    measurements = []
    for width in widths:
        trial = folder / f"width{width}"
        trial.mkdir()
        profile = trial / "sps.json"
        profile.write_text(
            json.dumps(
                {**base, "policy": SpsPolicy((width,) * 7, width * 8).to_dict()},
                indent=2,
            )
            + "\n"
        )
        with server(command, profile, trial, args.port, args.startup_timeout) as url:
            measured = measure(
                url,
                prefixes,
                sizes,
                args.repeats,
                args.max_new_tokens,
                trial / "requests.jsonl",
            )
        measurements.extend({**r, "width": width} for r in measured)
        (folder / "measurements.json").write_text(
            json.dumps(measurements, indent=2) + "\n"
        )
        print(json.dumps({"width": width, "measurements": measured}), flush=True)
    policy = select_policy(measurements)
    candidate = folder / "selected.json"
    candidate.write_text(
        json.dumps(
            {**base, "policy": policy.to_dict(), "measurements": measurements}, indent=2
        )
        + "\n"
    )
    # Test the combined policy, including its union of captured graph widths.
    with server(
        command, candidate, folder / "selected", args.port, args.startup_timeout
    ) as url:
        validation = measure(
            url,
            prefixes,
            sizes,
            1,
            args.max_new_tokens,
            folder / "selected" / "requests.jsonl",
        )
    output.write_text(
        json.dumps(
            {**json.loads(candidate.read_text()), "validation": validation}, indent=2
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "profile": str(output),
                "policy": policy.to_dict(),
                "validation": validation,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
