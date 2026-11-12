"""Opt-in Qwen3.8 serving regression at concurrency 32.

Set DSPINE_LARGE_BATCH_E2E=1, DSPINE_TARGET_MODEL and DSPINE_DRAFT_MODEL.
The runtime must have room for the BF16 target, draft and decode caches.
"""

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

import pytest
import requests

if os.environ.get("DSPINE_LARGE_BATCH_E2E") != "1":
    pytest.skip("set DSPINE_LARGE_BATCH_E2E=1", allow_module_level=True)

TARGET = os.environ["DSPINE_TARGET_MODEL"]
DRAFT = os.environ["DSPINE_DRAFT_MODEL"]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    output = tmp_path_factory.mktemp("large-batch-server")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        TARGET,
        "--speculative-algorithm",
        "DSPINE",
        "--speculative-draft-model-path",
        DRAFT,
        "--speculative-num-draft-tokens",
        "16",
        "--tp-size",
        "1",
        "--dtype",
        "bfloat16",
        "--attention-backend",
        "fa3",
        "--speculative-draft-attention-backend",
        "fa3",
        "--linear-attn-backend",
        "triton",
        "--enable-linear-replayssm-spec",
        "--mamba-ssm-dtype",
        "float32",
        "--mamba-radix-cache-strategy",
        "extra_buffer",
        "--max-mamba-cache-size",
        "256",
        "--context-length",
        "4096",
        "--mem-fraction-static",
        "0.75",
        "--max-running-requests",
        "32",
        "--cuda-graph-bs-decode",
        "1",
        "2",
        "4",
        "8",
        "16",
        "32",
        "--sampling-backend",
        "flashinfer",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--trust-remote-code",
    ]
    ipc = tempfile.mkdtemp(prefix="dspine-batch-", dir="/tmp")
    env = dict(os.environ, TMPDIR=ipc, SGLANG_PLUGINS="dspine")
    log_path = output / "server.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command, env=env, stdout=log, stderr=log, start_new_session=True
        )
        try:
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    pytest.fail(log_path.read_text()[-8000:])
                try:
                    if requests.get(url + "/health", timeout=5).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(2)
            else:
                pytest.fail("Server startup timed out: " + log_path.read_text()[-8000:])
            yield url, output
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)
            except ProcessLookupError:
                pass
            shutil.rmtree(ipc)


def test_large_greedy_sampled_and_mixed_batches(server):
    url, output = server
    greedy = {"temperature": 0.0}
    sampled = {"temperature": 1.0, "top_k": 20, "top_p": 0.95}
    general = {"temperature": 0.7, "top_k": 30, "top_p": 0.9, "min_p": 0.05}
    results = []
    for mode in ("greedy", "sampled", "mixed"):
        # Re-enter the graph path after a larger eager proposal, and exercise
        # batches that pad to a larger draft graph as well as exact graph sizes.
        for bs in (8, 9, 16, 17, 32, 8):
            params = []
            for i in range(bs):
                p = greedy if mode == "greedy" else sampled
                if mode == "mixed":
                    p = (greedy, sampled, general)[i % 3]
                params.append({**p, "max_new_tokens": 64, "ignore_eos": True})
            response = requests.post(
                url + "/generate",
                json={
                    "text": [
                        f"Write a Python function that sums a list. Example number {i}."
                        for i in range(bs)
                    ],
                    "sampling_params": params,
                },
                timeout=600,
            )
            response.raise_for_status()
            rows = response.json()
            assert len(rows) == bs
            for row in rows:
                assert row["text"]
                meta = row["meta_info"]
                assert meta["completion_tokens"] == 64
                assert 0 < meta["spec_verify_ct"] <= meta["completion_tokens"]
            results.append({"mode": mode, "batch_size": bs, "outputs": rows})
            (output / "outputs.json").write_text(json.dumps(results, indent=2) + "\n")
    # A scheduler trace is required to demonstrate real large decode batches;
    # sending 32 HTTP requests alone would also pass with a batch-eight server.
    trace = (output / "server.log").read_text()
    assert "sampler_capacity_bs=32" in trace
    assert "DSpine eager proposal table: batch=32 " in trace
