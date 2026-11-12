"""CPU control-flow regressions for the SPS release workflow."""

import contextlib
import hashlib
import importlib.util
import io
import json
import signal
import sys
import urllib.error
from pathlib import Path

import pytest


@pytest.fixture
def tuner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "tune_sps.py"
    spec = importlib.util.spec_from_file_location("tune_sps_release", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("command,expected", [
    (["--api-key", "client-secret", "--admin-api-key=admin-secret"], "client-secret"),
    (["--api-key=client-secret", "--admin-api-key", "admin-secret"], "client-secret"),
    (["--admin-api-key", "admin-secret"], None),
    (["--api-key=old-secret", "--api-key", "client-secret"], "client-secret"),
])
def test_native_normal_auth_precedence_and_recursive_redaction(tuner, command, expected):
    credentials = tuner.Credentials(command)
    assert credentials.headers("http://local/model_info") == (
        {"Authorization": "Bearer " + expected} if expected else {}
    )
    record = {"command": command, "nested": [{"api_key": "unknown-secret"},
        {"admin-api-key": "other-secret"}, {"Authorization": "Bearer unknown"}],
        "launch_command": " ".join(command)}
    encoded = json.dumps(credentials.redact(record))
    assert not any(secret in encoded for secret in credentials.secrets)
    assert "unknown-secret" not in encoded and "other-secret" not in encoded


@pytest.mark.parametrize("key", ["dspine", "sampled", "1", "table"])
def test_short_public_word_keys_preserve_loadable_profile_and_calibration(tuner, tmp_path, key):
    calibration = json.loads((Path(tuner.__file__).parents[1] / "sglang_dspine" /
        "verification" / "calibration.json").read_text())
    original = {"format": "dspine-sps-v2", "hardware": {},
        "policy": {"verify_widths": {}, "token_budgets": {}},
        "mode_policies": {"sampled": {"verify_widths": {}, "token_budgets": {"1": 8}}},
        "mode": "sampled", "requests_sha256": "012345", "confidence_calibration": calibration,
        "command": ["sglang", "serve", "--api-key", key], "nested": {"api_key": key}}
    clean = tuner.Credentials(original["command"]).redact(original)
    for name in ("format", "mode", "requests_sha256", "confidence_calibration"):
        assert clean[name] == original[name]
    assert clean["command"][-1] == clean["nested"]["api_key"] == "[REDACTED]"
    path = tmp_path / "sps.json"
    path.write_text(json.dumps(clean))
    _, modes = tuner.SpsPolicy.load_with_modes(path, {})
    assert modes["sampled"].budget(1) == 8
    assert set(tuner.load_calibration_tables(clean["confidence_calibration"])) == {"greedy", "sampled"}


@pytest.mark.parametrize("timeout", [None, 3600.0])
def test_generation_timeout_and_bearer_header_reach_urlopen(tuner, monkeypatch, timeout):
    monkeypatch.setattr(tuner, "_credentials", tuner.Credentials(["--api-key=client-secret"]))
    observed = []

    @contextlib.contextmanager
    def urlopen(request, timeout):
        observed.append((dict(request.header_items()), timeout))
        yield io.StringIO("{}")

    monkeypatch.setattr(tuner.urllib.request, "urlopen", urlopen)
    tuner.request("http://local/generate", {"input_ids": [1]}, timeout=timeout)
    assert observed == [({"Content-type": "application/json", "Authorization": "Bearer client-secret"}, timeout)]


@pytest.mark.parametrize("status", [401, 403])
def test_startup_auth_failure_is_immediate_and_native_logs_are_redacted(tuner, monkeypatch, tmp_path, status):
    secret = "synthetic-quote-\"-secret"
    calls, signals = [], []

    class Process:
        pid = 12345
        returncode = None
        stdout = io.StringIO(json.dumps({"api_key": secret}) + "\n" + secret + "\n")

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    def popen(argv, **kwargs):
        assert secret in argv  # The actual subprocess keeps the real credential.
        assert kwargs["stdout"] == tuner.subprocess.PIPE
        return Process()

    def urlopen(request, timeout):
        calls.append((dict(request.header_items()), timeout))
        raise urllib.error.HTTPError(request.full_url, status, "auth error", {}, io.BytesIO())

    monkeypatch.setattr(tuner.subprocess, "Popen", popen)
    monkeypatch.setattr(tuner.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(tuner.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(tuner.time, "sleep", lambda delay: pytest.fail("Authentication must not retry"))
    previous = tuner._credentials
    with pytest.raises(tuner.AuthenticationError, match=f"HTTP {status}"):
        with tuner.server(["sglang", "serve", "--api-key", secret], tmp_path / "sps.json", tmp_path, 0, 1800):
            pytest.fail("Readiness must fail")
    assert calls == [({"Content-type": "application/json", "Authorization": "Bearer " + secret}, 2)]
    assert signals == [(12345, signal.SIGTERM), (12345, signal.SIGKILL)]
    assert tuner._credentials is previous
    for path in (tmp_path / "command.json", tmp_path / "server.log"):
        assert secret not in path.read_text()
        assert json.dumps(secret)[1:-1] not in path.read_text()


def test_turnkey_rejects_algorithm_word_without_cli_flag(tuner, monkeypatch, tmp_path):
    path = tmp_path / "requests.jsonl"
    path.write_text(json.dumps({"input_ids": [1]}) + "\n")
    monkeypatch.setattr(sys, "argv", [str(tuner.__file__), "--requests", str(path),
        "--output", str(tmp_path / "profile.json"), "--batch-sizes", "8", "--",
        "sglang", "serve", "DSPINE"])
    with pytest.raises(SystemExit) as error:
        tuner.main()
    assert error.value.code == 2


def test_flush_text_is_allowed_but_generation_remains_json(tuner, monkeypatch):
    @contextlib.contextmanager
    def urlopen(request, timeout):
        yield io.StringIO("Cache flushed successfully!")
    monkeypatch.setattr(tuner.urllib.request, "urlopen", urlopen)
    assert tuner.request("http://local/flush_cache", {}, allow_text=True)["body"].startswith("Cache flushed")
    with pytest.raises(ValueError):
        tuner.request("http://local/generate", {"input_ids": [1]})
