"""SPS measurements belong to a TP topology and backend configuration."""

import json

import pytest
from sglang_dspine.verification import policy as _policy


def profile(tmp_path, execution=None):
    data = {
        "format": "dspine-sps-v2",
        "hardware": {},
        "policy": {"token_budgets": {"3": 32}},
    }
    if execution is not None:
        data["execution"] = execution
    path = tmp_path / "sps.json"
    path.write_text(json.dumps(data))
    return path


def test_legacy_profile_is_only_implicitly_single_rank(tmp_path):
    path = profile(tmp_path)
    assert (
        _policy.SpsPolicy.load(path, {}, _policy.execution_identity(1, 1)).budget(3)
        == 32
    )
    with pytest.raises(ValueError, match="no TP layout"):
        _policy.SpsPolicy.load(path, {}, _policy.execution_identity(2, 2))


@pytest.mark.parametrize(
    "field,value",
    [
        ("tp_size", 4),
        ("draft_tp_size", 1),
        ("attention_backend", "triton"),
        ("draft_attention_backend", "fa3"),
    ],
)
def test_profile_rejects_other_execution_layout(tmp_path, field, value):
    execution = _policy.execution_identity(2, 2, "fa3", "triton")
    path = profile(tmp_path, execution)
    expected = {**execution, field: value}
    with pytest.raises(ValueError, match="execution layout differs"):
        _policy.SpsPolicy.load(path, {}, expected)
    assert _policy.SpsPolicy.load(path, {}, execution).budget(3) == 32


def test_probe_can_record_layout_before_backend_resolution(tmp_path):
    probe = _policy.execution_identity(2, 2)
    path = profile(tmp_path, probe)
    resolved = _policy.execution_identity(2, 2, "fa3", "triton")
    assert _policy.SpsPolicy.load(path, {}, resolved).budget(3) == 32
