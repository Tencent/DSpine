"""CPU checks for censored outcomes, identifiable fits and runtime parity."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sglang_dspine.verification.calibration import ConfidenceCalibrator, FORMAT


@pytest.fixture(scope="module")
def fitting():
    path = Path(__file__).resolve().parents[1] / "scripts" / "fit_calibration.py"
    spec = importlib.util.spec_from_file_location("fit_confidence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic(n=1000, gamma=3, seed=0):
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.65, 0.99, (n, gamma))
    # The proxy is strongly overconfident, including at the first depth.
    probability = 1 / (1 + np.exp(-(np.log(raw / (1 - raw)) * 0.6 - 1.9)))
    accepted = np.cumprod(rng.random((n, gamma)) < probability, axis=1).sum(axis=1)
    return raw, accepted


def test_first_failure_censors_only_conditional_labels(fitting):
    labels, observed = fitting.outcomes([0, 1, 3], 3)
    np.testing.assert_array_equal(labels, [[0, 0, 0], [1, 0, 0], [1, 1, 1]])
    np.testing.assert_array_equal(observed, [[1, 0, 0], [1, 1, 0], [1, 1, 1]])
    metrics = fitting.evaluate({"kind": "identity", "gamma": 3}, np.full((3, 3), 0.5), np.array([0, 1, 3]))
    assert metrics["conditional"]["count"] == 6
    assert metrics["prefix"]["count"] == 9
    assert metrics["accepted_length"]["predicted"] == pytest.approx(0.5 + 0.25 + 0.125)


def test_legacy_mapping_reproduces_existing_smoothing(fitting):
    data = {"success": [[0, 8], [0, 0]], "count": [[4, 10], [0, 0]]}
    model = fitting.legacy_model(data)
    prior = (8 + 8 * 0.85) / (14 + 8)
    assert model["probabilities"][0] == pytest.approx([12 * prior / 16, (8 + 12 * prior) / 22])
    assert model["probabilities"][1] == pytest.approx([0.85, 0.85])


def test_sparse_and_missing_depths_stay_finite_and_monotonic(fitting):
    raw = np.array([[0, 0.8, 1], [1, 0.6, 0], [0.5, 0.9, 0.4]])
    accepted = np.zeros(3, dtype=int)
    for model in (fitting.fit_sts(raw, accepted, 2), fitting.fit_isotonic(raw, accepted, 16, 2), fitting.fit_joint(raw, accepted, 2)):
        prediction = fitting.predict(model, raw)
        assert np.isfinite(prediction).all()
        assert np.all((prediction >= 0) & (prediction <= 1))
        line = np.broadcast_to(np.linspace(0, 1, 101)[:, None], (101, 3))
        assert np.all(np.diff(fitting.predict(model, line), axis=0) >= 0)
    sts = fitting.fit_sts(raw, accepted, 2)
    assert sts["temperature"][1:] == [1, 1]
    joint = fitting.fit_joint(raw, accepted, 2)
    assert joint["scale"][1:] == [1, 1] and joint["offset"][1:] == [0, 0]
    empty = fitting.fit_joint(raw, accepted, 10)
    assert empty["kind"] == "identity"


@pytest.mark.parametrize("loss", ["brier", "bce"])
def test_joint_analytical_gradient_matches_finite_differences(fitting, loss):
    raw, accepted = synthetic(11, seed=8)
    labels, _ = fitting.outcomes(accepted, 3)
    logits = np.log(raw / (1 - raw))
    active = np.array([True, False, True])
    parameters = np.array([0.2, -0.3, -0.8])
    _, analytical = fitting.joint_objective(parameters, logits, labels, active, loss)
    numeric = []
    for i in range(len(parameters)):
        delta = np.zeros_like(parameters)
        delta[i] = 1e-5
        hi = fitting.joint_objective(parameters + delta, logits, labels, active, loss)[0]
        lo = fitting.joint_objective(parameters - delta, logits, labels, active, loss)[0]
        numeric.append((hi - lo) / 2e-5)
    np.testing.assert_allclose(analytical, numeric, atol=1e-7, rtol=1e-5)


def test_joint_calibrates_below_half_and_improves_independent_prefix_brier(fitting):
    raw, accepted = synthetic(1500)
    test_raw, test_accepted = synthetic(1000, seed=19)
    model = fitting.fit_joint(raw, accepted, 20)
    exact = fitting.predict(model, test_raw)
    assert np.any(exact[:, 0] < 0.5)
    identity = fitting.evaluate({"kind": "identity", "gamma": 3}, test_raw, test_accepted)
    fitted = fitting.evaluate(model, test_raw, test_accepted)
    assert fitted["prefix"]["brier"] < identity["prefix"]["brier"] - 0.1
    assert len(model["scale"]) + 1 == 4  # shared alpha plus 3 offsets
    assert len(set(model["scale"])) == 1
    for d in range(3):
        assert fitted["conditional_by_depth"][d]["auc"] == pytest.approx(identity["conditional_by_depth"][d]["auc"])


def test_runtime_numpy_parity_boundaries_and_prefix_products(fitting):
    raw, accepted = synthetic(100)
    models = [fitting.fit_joint(raw, accepted, 3), fitting.fit_sts(raw, accepted, 3), fitting.fit_isotonic(raw, accepted, 16, 3), {"kind": "identity", "gamma": 3}]
    values = np.array([[0, 0.5, 1], [1, 0, 0.25], [0.8, 0.99, 0.6]], dtype=np.float32)
    for model in models:
        calibrator = ConfidenceCalibrator(model)
        expected = fitting.predict(model, values)
        # Float32's representation of 1-eps differs slightly at the logit edge.
        np.testing.assert_allclose(calibrator(torch.tensor(values)).numpy(), expected, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(calibrator.prefix_survival(torch.tensor(values)).numpy(), np.cumprod(expected, axis=1), atol=1e-5, rtol=1e-5)
        assert torch.all(torch.diff(calibrator.prefix_survival(torch.tensor(values)), dim=-1) <= 0)


def test_finer_lookup_reduces_quantization_error(fitting):
    raw, accepted = synthetic(200)
    model = fitting.fit_joint(raw, accepted, 5)
    probe = np.broadcast_to(np.linspace(0.001, 0.999, 2001)[:, None], (2001, 3))
    errors = [np.abs(fitting.predict(fitting.lookup_model(model, bins), probe) - fitting.predict(model, probe)).mean() for bins in (16, 256, 1024)]
    assert errors[2] < errors[1] < errors[0]
    artifact = {"format": FORMAT, "families": {"sampled": {"model": model, "lookup_tables": {"256": fitting.lookup_model(model, 256)}}}}
    runtime = ConfidenceCalibrator.from_artifact(artifact, "sampled", lookup_bins=256)
    assert runtime.probabilities.shape == (3, 256)


def test_logit_lookup_preserves_endpoints_boundaries_and_saturated_prefixes(fitting):
    model = {"kind": "logit", "gamma": 3, "scale": [0.216] * 3, "offset": [-0.3, 0.1, 0.5], "eps": fitting.EPS}
    table = fitting.lookup_model(model, 1024, "logit")
    boundaries = np.asarray(table["boundaries"])
    assert len(boundaries) == 1023 and np.all(np.diff(boundaries) > 0)
    assert table["representatives"][0] == 0 and table["representatives"][-1] == 1
    runtime = ConfidenceCalibrator(table)
    edge = boundaries[::31].astype(np.float32)
    probe = np.r_[0, 1, edge, np.nextafter(edge, np.float32(0)), np.nextafter(edge, np.float32(1))].astype(np.float32)
    raw = np.broadcast_to(probe[:, None], (len(probe), 3)).copy()
    np.testing.assert_allclose(runtime(torch.tensor(raw)).numpy(), fitting.predict(table, raw), atol=1e-7)
    np.testing.assert_allclose(fitting.predict(table, raw[:2]), fitting.predict(model, raw[:2]), atol=1e-12)
    generator = np.random.default_rng(42)
    saturated = generator.uniform(0.1, 1.0, (1000, 3))
    saturated[generator.random(saturated.shape) < 0.6] = 1
    exact = np.cumprod(fitting.predict(model, saturated), -1)
    approx = np.cumprod(fitting.predict(table, saturated), -1)
    assert np.abs(approx - exact).mean() < 0.002
    assert runtime.boundaries.dtype == torch.float64


def test_conversation_split_blocks_leakage_and_partial_verification(tmp_path, fitting):
    common = {"confidence": [0.8, 0.9], "accepted": 1, "mode": "sampled", "conversation_id": "same", "sampling_params": {"temperature": 0.7}, "benchmark": "example"}
    path = tmp_path / "rounds.jsonl"
    path.write_text("\n".join(json.dumps({**common, "split": split}) for split in ("calib", "test")))
    with pytest.raises(ValueError, match="multiple splits"):
        fitting.load_rounds(path)
    path.write_text(json.dumps({**common, "verify_len": 2}))
    with pytest.raises(ValueError, match="full-block"):
        fitting.load_rounds(path)
    path.write_text("\n".join(json.dumps(common) for _ in range(4)))
    assert len({r["split"] for r in fitting.load_rounds(path, 4)}) == 1
    assert fitting.load_rounds(path, 4) == fitting.load_rounds(path, 4)


def test_test_labels_never_change_fitted_model_or_selection(fitting):
    rows = []
    for split, seed in (("calib", 0), ("val", 1), ("test", 2)):
        raw, accepted = synthetic(100, seed=seed)
        rows.extend({"confidence": c.tolist(), "accepted": int(a), "conversation_id": f"{split}-{i}", "mode": "greedy", "split": split, "benchmark": "synthetic", "cohort_id": f"micro-{i}"} for i, (c, a) in enumerate(zip(raw, accepted)))
    first, report = fitting.run(rows, min_observed=5)
    assert report["families"]["greedy"]["test"] == {}
    assert report["families"]["greedy"]["test_scope"] == "withheld"
    assert set(report["families"]["greedy"]["quantization"]) == {"val"}
    assert set(report["families"]["greedy"]["selected_by_group"]["val"]) == {"benchmark"}
    modified = [{**r, "accepted": 3 - r["accepted"]} if r["split"] == "test" else r for r in rows]
    second, _ = fitting.run(modified, min_observed=5)
    assert first["families"]["greedy"]["model"] == second["families"]["greedy"]["model"]
    assert first["families"]["greedy"]["selected"] == second["families"]["greedy"]["selected"]


def test_auc_handles_ties_and_no_positive_class(fitting):
    assert fitting.auc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5
    assert fitting.auc([0.2, 0.3], [0, 0]) is None


def test_sampled_actual_chain_source_is_distinct_from_legacy_proxy(tmp_path, fitting):
    path = tmp_path / "sampled.jsonl"
    rows = [{"confidence": [0.6, 0.7], "greedy_confidence": [0.9, 0.95], "accepted": 0, "conversation_id": split, "mode": "sampled", "split": split} for split in ("calibration", "validation")]
    path.write_text('\n'.join(json.dumps(row) for row in rows))
    actual_rows = fitting.load_rounds(path)
    actual, _ = fitting.run(actual_rows)
    assert actual["confidence_source"] == "pre_token_max_q_actual_sampled_chain"
    assert actual["confidence_sources"] == {"sampled": fitting.SAMPLED_SOURCE}
    assert actual["families"]["sampled"]["evidence"]["confidence_source"] == fitting.SAMPLED_SOURCE
    legacy_rows = [{key: value for key, value in row.items() if key not in ("greedy_confidence", "confidence_source")} for row in actual_rows]
    legacy, _ = fitting.run(legacy_rows)
    assert legacy["confidence_source"] == fitting.GREEDY_SOURCE
    assert fitting.confidence_source({"mode": "sampled", "confidence_source": fitting.SAMPLED_SOURCE}) == fitting.SAMPLED_SOURCE
    with pytest.raises(ValueError, match="mixes different confidence sources"):
        fitting.run([actual_rows[0], legacy_rows[1]])


def test_legacy_comparison_reads_greedy_feature_not_new_sampled_chain(fitting):
    rows = [{"confidence": [0.1, 0.2], "greedy_confidence": [0.9, 0.95], "confidence_source": fitting.SAMPLED_SOURCE, "accepted": 0, "conversation_id": split, "mode": "sampled", "split": split} for split in ("calib", "val", "test")]
    legacy = {"success": [[0, 20], [0, 20]], "count": [[20, 20], [20, 20]]}
    _, report = fitting.run(rows, legacy=legacy, evaluate_test=True)
    expected = fitting.evaluate(fitting.legacy_model(legacy), np.array([[0.9, 0.95]]), np.array([0]))
    wrong = fitting.evaluate(fitting.legacy_model(legacy), np.array([[0.1, 0.2]]), np.array([0]))
    assert report["families"]["sampled"]["validation"]["legacy_map"] == expected
    assert report["families"]["sampled"]["test"]["legacy_map"] == expected
    assert expected["prefix"]["brier"] != wrong["prefix"]["brier"]
    with pytest.raises(ValueError, match="requires greedy_confidence"):
        fitting.legacy_features([{k: v for k, v in rows[0].items() if k != "greedy_confidence"}])


def test_pretruncation_source_keeps_filtered_feature_diagnostic_only(tmp_path, fitting):
    rows = [{"confidence": [0.85, 0.99], "greedy_confidence": [0.9, 0.95], "filtered_confidence": [1.0, 1.0], "confidence_source": fitting.SAMPLED_PROBABILITY_SOURCE, "accepted": 0, "conversation_id": split, "mode": "sampled", "split": split} for split in ("calib", "val")]
    path = tmp_path / "preprob.jsonl"
    path.write_text('\n'.join(json.dumps(row) for row in rows))
    loaded = fitting.load_rounds(path)
    raw, _ = fitting.arrays(loaded)
    np.testing.assert_array_equal(raw, [[0.85, 0.99]] * 2)
    artifact, report = fitting.run(loaded)
    assert artifact["families"]["sampled"]["confidence_source"] == "pre_token_max_probability_actual_sampled_chain"
    assert report["families"]["sampled"]["test"] == {}
    rows[0]["filtered_confidence"] = [1.0]
    path.write_text('\n'.join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="filtered_confidence"):
        fitting.load_rounds(path)


def test_runtime_rejects_invalid_fitted_parameters():
    with pytest.raises(ValueError, match="positive"):
        ConfidenceCalibrator({"kind": "temperature", "gamma": 1, "temperature": [0]})
    with pytest.raises(ValueError, match="decreases"):
        ConfidenceCalibrator({"kind": "table", "gamma": 1, "probabilities": [[0.8, 0.2]], "monotonic": True})
