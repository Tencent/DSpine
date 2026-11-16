"""The real SPS CLI/search retains the deployment's tensor-parallel layout."""

import contextlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "tp_args,tp_size",
    [
        (["--tp-size", "2"], 2),
        (["--tp", "2"], 2),
        (["--tensor-parallel-size", "2"], 2),
        (["--tp-size", "2", "--tp", "4"], 4),
        (["--tp=4", "--tp-size=2"], 2),
        (["--tp-size", "2", "--tensor-parallel-size=8", "--tp", "4"], 4),
        (["--tp-size", "2", "--tp=4", "--tensor-parallel-size=8"], 8),
        (["--tp"], None),
        (["--tp-size", "--tp=4"], None),
        (["--tensor-parallel-size="], None),
    ],
)
def test_search_profiles_the_cli_and_resolved_parallel_layout(
    tmp_path, monkeypatch, tp_args, tp_size
):
    source = Path(__file__).resolve().parents[1] / "scripts/tune_sps.py"
    spec = importlib.util.spec_from_file_location("dspine_tp_tuner_under_test", source)
    tuner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tuner)
    expected = tuner.execution_identity(tp_size or 1, tp_size or 1, "fa3", "triton")

    def write_rows(name, tasks):
        path = tmp_path / name
        rows = [
            {
                "task_id": task,
                "input_ids": [task],
                "sampling_params": {"temperature": 0},
            }
            for task in tasks
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    training = write_rows("tune.jsonl", range(1, 7))
    holdout = write_rows("holdout.jsonl", range(20, 23))
    destination = tmp_path / "sps.json"

    @contextlib.contextmanager
    def server(_command, profile, *_args):
        assert tp_size is not None, "Malformed TP arguments must fail before serving"
        assert json.loads(profile.read_text())["execution"] == expected
        yield "http://cpu-fixture"

    def request(*_args, **_kwargs):
        return {
            "speculative_algorithm": "DSPINE",
            "speculative_num_draft_tokens": 16,
            "tp_size": tp_size,
            "attention_backend": "fa3",
            "speculative_draft_attention_backend": "triton",
            "cuda_graph_config": {"decode": {"backend": "full", "bs": [1, 2, 3]}},
        }

    def trial(_command, profile, _rows, _bs, repeat, count, *_args, **_kwargs):
        assert profile["execution"] == expected
        return {
            "status": "PASS",
            "repeat": repeat,
            "requests": count,
            "wall_seconds": 2.0,
            "output_tokens": 200,
            "tokens_per_second": 100.0,
            "verify_calls": count,
            "jobs_sha256": f"{repeat}-{count}",
        }

    monkeypatch.setattr(tuner, "hardware_identity", lambda _device: {})
    monkeypatch.setattr(tuner, "server", server)
    monkeypatch.setattr(tuner, "request", request)
    monkeypatch.setattr(tuner, "run_profile_trials", trial)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(source),
            "--requests",
            str(training),
            "--holdout-requests",
            str(holdout),
            "--output",
            str(destination),
            "--batch-sizes",
            "3",
            "--widths",
            "8",
            "16",
            "--repeats",
            "2",
            "--min-seconds",
            "1",
            "--min-requests",
            "4",
            "--",
            "sglang",
            "serve",
            "--speculative-algorithm",
            "DSPINE",
            *tp_args,
            "--attention-backend",
            "fa3",
            "--speculative-draft-attention-backend",
            "triton",
        ],
    )
    if tp_size is None:
        with pytest.raises(ValueError):
            tuner.main()
        assert not destination.exists()
        assert not destination.with_name("sps.runs").exists()
        return
    tuner.main()
    result = json.loads(destination.read_text())
    assert result["execution"] == expected
    tuner.SpsPolicy.load(destination, {}, expected)
    with pytest.raises(ValueError, match="execution layout differs"):
        tuner.SpsPolicy.load(
            destination,
            {},
            tuner.execution_identity(
                2 if tp_size == 4 else 4, 2 if tp_size == 4 else 4, "fa3", "triton"
            ),
        )
