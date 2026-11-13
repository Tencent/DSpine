#!/usr/bin/env python3
"""Fit confidence maps on full-block rounds and compare on conversation holdouts.

Example: python scripts/fit_calibration.py rounds.jsonl --output calibration.json --report metrics.json

Required JSONL fields: confidence, accepted, conversation_id, mode. Optional
split is calib/val/test; if absent for every row, split whole conversations
deterministically. Other collector fields are accepted. All rounds must verify
the full block. accepted excludes anchor/bonus. Conditional outcomes after the
first failure are censored; prefix-survival outcomes remain known at all depths.
No target distribution or target token probabilities are required.

Sampled confidence is recorded before drawing the current token on the actual
sampled prefix. Its source identifies either the maximum temperature-softmax
probability before truncation or the older filtered-q maximum. Optional
greedy_confidence and filtered_confidence retain diagnostic features; fitting
uses confidence only. Prefer an explicit per-row confidence_source; older
confidence-only rows retain the legacy proxy source.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


FORMAT = "dspine-confidence-calibration-v1"
EPS = 1e-6
SPLITS = {"calib": "calib", "calibration": "calib", "val": "val", "validation": "val", "test": "test"}
GREEDY_SOURCE = "greedy_chain_top1_top16_softmax"
SAMPLED_SOURCE = "pre_token_max_q_actual_sampled_chain"
SAMPLED_PROBABILITY_SOURCE = "pre_token_max_probability_actual_sampled_chain"


def confidence_source(row):
    source = row.get("confidence_source")
    if source is None:
        source = SAMPLED_SOURCE if row["mode"] == "sampled" and "greedy_confidence" in row else GREEDY_SOURCE
    if not isinstance(source, str) or not source:
        raise ValueError("confidence_source must be a nonempty string")
    return source


def outcomes(accepted, gamma):
    """Return prefix labels and observed-conditional mask, both [round, depth]."""
    accepted = np.asarray(accepted)
    depth = np.arange(gamma)[None, :]
    return (depth < accepted[:, None]).astype(float), depth <= accepted[:, None]


def load_rounds(path, seed=0):
    rows, split_by_conversation, explicit = [], {}, set()
    gamma = None
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        confidence = np.asarray(row["confidence"], dtype=float)
        if confidence.ndim != 1 or len(confidence) < 1 or not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
            raise ValueError(f"line {line_number}: confidence must be finite probabilities")
        gamma = len(confidence) if gamma is None else gamma
        if len(confidence) != gamma:
            raise ValueError("confidence depth differs across rounds")
        accepted = row["accepted"]
        if type(accepted) is not int or not 0 <= accepted <= gamma:
            raise ValueError("accepted must be an integer excluding anchor, in [0, gamma]")
        if row["mode"] not in ("greedy", "sampled"):
            raise ValueError("mode must be greedy or sampled")
        row["confidence_source"] = confidence_source(row)
        for field in ("greedy_confidence", "filtered_confidence"):
            if field in row:
                diagnostic = np.asarray(row[field], dtype=float)
                if diagnostic.shape != confidence.shape or not np.isfinite(diagnostic).all() or np.any((diagnostic < 0) | (diagnostic > 1)):
                    raise ValueError(f"{field} must match confidence depth and contain finite probabilities")
        if not isinstance(row["conversation_id"], (str, int)) or isinstance(row["conversation_id"], bool):
            raise ValueError("conversation_id must be a string or integer")
        if row.get("verify_len", gamma + 1) != gamma + 1:
            raise ValueError("calibration requires full-block rounds")
        conversation = str(row["conversation_id"])
        explicit.add("split" in row)
        split = row.get("split")
        if "split" in row:
            if split not in SPLITS:
                raise ValueError("split must be calibration/calib, validation/val, or test")
            split = row["split"] = SPLITS[split]
        if conversation in split_by_conversation and split_by_conversation[conversation] != split:
            raise ValueError("conversation appears in multiple splits")
        split_by_conversation[conversation] = split
        rows.append(row)
    if not rows:
        raise ValueError("no calibration rounds")
    if len(explicit) != 1:
        raise ValueError("provide split on every round or on no rounds")
    if explicit == {False}:
        for conversation in split_by_conversation:
            digest = hashlib.sha256(f"{seed}:{conversation}".encode()).digest()
            u = int.from_bytes(digest[:8], "big") / 2**64
            split_by_conversation[conversation] = "calib" if u < 0.6 else "val" if u < 0.8 else "test"
        for row in rows:
            row["split"] = split_by_conversation[str(row["conversation_id"])]
    return rows


def arrays(rows):
    return np.asarray([r["confidence"] for r in rows], dtype=float), np.asarray([r["accepted"] for r in rows], dtype=int)


def legacy_features(rows):
    """The old map consumes the old feature, even alongside actual-chain q."""
    if any("greedy_confidence" not in row and confidence_source(row) != GREEDY_SOURCE for row in rows):
        raise ValueError("legacy comparison of actual-chain features requires greedy_confidence")
    return np.asarray([row.get("greedy_confidence", row["confidence"]) for row in rows], dtype=float)


def sigmoid(z):
    # Stable at both logit bounds without importing a fitting library at runtime.
    return np.exp(-np.logaddexp(0, -z))


def predict(model, raw):
    raw = np.asarray(raw, dtype=float)
    kind = model["kind"]
    if raw.shape[-1] != model["gamma"]:
        raise ValueError("confidence depth mismatch")
    if kind == "identity":
        return np.clip(raw, 0, 1)
    if kind == "table":
        table = np.asarray(model["probabilities"])
        index = (np.searchsorted(model["boundaries"], raw, side="left") if "boundaries" in model
                 else np.clip((raw * table.shape[-1]).astype(int), 0, table.shape[-1] - 1))
        return table[np.arange(raw.shape[-1]), index]
    p = np.clip(raw, model.get("eps", EPS), 1 - model.get("eps", EPS))
    logits = np.log(p) - np.log1p(-p)
    if kind == "temperature":
        return sigmoid(logits / np.asarray(model["temperature"]))
    if kind == "logit":
        return sigmoid(logits * np.asarray(model["scale"]) + np.asarray(model["offset"]))
    raise ValueError(f"unknown calibration model {kind}")


def ece(probability, labels, bins=15):
    if len(probability) == 0:
        return None
    index = np.clip((np.asarray(probability) * bins).astype(int), 0, bins - 1)
    counts = np.bincount(index, minlength=bins)
    predicted = np.bincount(index, weights=probability, minlength=bins)
    observed = np.bincount(index, weights=labels, minlength=bins)
    return float(np.abs(predicted - observed).sum() / counts.sum())


def auc(probability, labels):
    """Tie-aware rank AUC; absent classes yield None, never a fabricated 0.5."""
    probability, labels = np.asarray(probability), np.asarray(labels)
    positive, negative = int(labels.sum()), int(len(labels) - labels.sum())
    if not positive or not negative:
        return None
    order = np.argsort(probability, kind="stable")
    sorted_p = probability[order]
    starts = np.r_[0, np.flatnonzero(sorted_p[1:] != sorted_p[:-1]) + 1]
    ends = np.r_[starts[1:], len(sorted_p)]
    ranks = np.repeat((starts + ends + 1) / 2, ends - starts)
    rank_sum = float(ranks[labels[order] == 1].sum())
    return (rank_sum - positive * (positive + 1) / 2) / (positive * negative)


def scores(probability, labels, bins=15):
    probability, labels = np.asarray(probability), np.asarray(labels)
    if not len(probability):
        return {"count": 0, "ece": None, "brier": None, "bce": None, "auc": None, "predicted": None, "observed": None}
    p = np.clip(probability, EPS, 1 - EPS)
    return {
        "count": len(labels), "ece": ece(probability, labels, bins),
        "brier": float(np.mean((probability - labels)**2)),
        "bce": float(-np.mean(labels * np.log(p) + (1 - labels) * np.log1p(-p))),
        "auc": auc(probability, labels), "predicted": float(probability.mean()), "observed": float(labels.mean()),
    }


def evaluate(model, raw, accepted, ece_bins=15):
    conditional = predict(model, raw)
    prefix = np.cumprod(conditional, axis=-1)
    labels, observed = outcomes(accepted, raw.shape[1])
    error = prefix.sum(axis=1) - accepted
    return {
        "rounds": len(raw),
        "conditional": scores(conditional[observed], labels[observed], ece_bins),
        "prefix": scores(prefix.ravel(), labels.ravel(), ece_bins),
        "conditional_by_depth": [scores(conditional[observed[:, d], d], labels[observed[:, d], d], ece_bins) for d in range(raw.shape[1])],
        "prefix_by_depth": [scores(prefix[:, d], labels[:, d], ece_bins) for d in range(raw.shape[1])],
        "accepted_length": {"predicted": float(prefix.sum(axis=1).mean()), "observed": float(accepted.mean()), "bias": float(error.mean()), "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.mean(error**2)))},
    }


def bin_counts(raw, accepted, bins):
    labels, observed = outcomes(accepted, raw.shape[1])
    index = np.clip((raw * bins).astype(int), 0, bins - 1)
    count, success = [], []
    for d in range(raw.shape[1]):
        count.append(np.bincount(index[observed[:, d], d], minlength=bins))
        success.append(np.bincount(index[observed[:, d], d], weights=labels[observed[:, d], d], minlength=bins))
    return np.asarray(success), np.asarray(count)


def legacy_model(data):
    """Exactly the existing SPS success/count smoothing, including empty bins."""
    success, count = np.asarray(data["success"], dtype=float), np.asarray(data["count"], dtype=float)
    if success.shape != count.shape or success.ndim != 2 or success.shape[1] < 2 or np.any(success < 0) or np.any(count < success) or not np.isfinite(success).all() or not np.isfinite(count).all():
        raise ValueError("invalid legacy success/count table")
    prior = (success.sum(axis=1) + 8 * 0.85) / (count.sum(axis=1) + 8)
    table = np.clip((success + 12 * prior[:, None]) / (count + 12), 0.05, 0.999)
    return {"kind": "table", "gamma": len(table), "probabilities": table.tolist(), "monotonic": False}


def pava(values, weights):
    """Weighted least-squares projection onto nondecreasing bin probabilities."""
    blocks = []
    for i, (value, weight) in enumerate(zip(values, weights)):
        blocks.append([i, i + 1, float(value * weight), float(weight)])
        while len(blocks) > 1 and blocks[-2][2] / blocks[-2][3] > blocks[-1][2] / blocks[-1][3]:
            right, left = blocks.pop(), blocks.pop()
            blocks.append([left[0], right[1], left[2] + right[2], left[3] + right[3]])
    result = np.empty(len(values))
    for start, end, total, weight in blocks:
        result[start:end] = total / weight
    return result


def fit_isotonic(raw, accepted, bins=16, min_observed=20):
    success, count = bin_counts(raw, accepted, bins)
    centers = (np.arange(bins) + 0.5) / bins
    table = np.tile(centers, (raw.shape[1], 1))
    for d in range(raw.shape[1]):
        if count[d].sum() >= min_observed:
            # Half a pseudo-observation per bin provides finite, monotonic
            # interpolation in empty bins without inventing target labels.
            weight = count[d] + 0.5
            table[d] = pava((success[d] + 0.5 * centers) / weight, weight)
    return {"kind": "table", "gamma": raw.shape[1], "probabilities": table.tolist(), "monotonic": True}


def fit_sts(raw, accepted, min_observed=20, ece_bins=15, grid=None):
    """DSpark-style sequential temperature search on cumulative-prefix ECE."""
    gamma = raw.shape[1]
    labels, observed = outcomes(accepted, gamma)
    p = np.clip(raw, EPS, 1 - EPS)
    logits = np.log(p) - np.log1p(-p)
    grid = np.unique(np.r_[np.geomspace(0.05, 20, 81), 1.0] if grid is None else np.r_[grid, 1.0])
    if not np.isfinite(grid).all() or np.any(grid <= 0):
        raise ValueError("STS temperatures must be finite and positive")
    temperatures, prefix = np.ones(gamma), np.ones(len(raw))
    for d in range(gamma):
        if observed[:, d].sum() >= min_observed:
            losses = [ece(prefix * sigmoid(logits[:, d] / t), labels[:, d], ece_bins) for t in grid]
            best_loss = min(losses)
            # Among numerical ties, prefer the map closest to identity.
            best = min((i for i, loss in enumerate(losses) if loss <= best_loss + 1e-12), key=lambda i: abs(math.log(grid[i])))
            temperatures[d] = grid[best]
        prefix *= sigmoid(logits[:, d] / temperatures[d])
    return {"kind": "temperature", "gamma": gamma, "temperature": temperatures.tolist(), "eps": EPS}


def joint_objective(parameters, logits, labels, active, loss="brier", regularization=1e-4):
    """Joint prefix loss and analytical gradient for shared log-alpha + offsets."""
    alpha = math.exp(parameters[0])
    offset = np.zeros(logits.shape[1])
    offset[active] = parameters[1:]
    scale = np.where(active, alpha, 1.0)
    log_prefix = np.cumsum(-np.logaddexp(0, -(logits * scale + offset)), axis=1)
    prefix = np.exp(log_prefix)
    if loss == "brier":
        value = np.mean((prefix - labels)**2)
        derivative = 2 * (prefix - labels) * prefix / prefix.size
    elif loss == "bce":
        # log(1-exp(log_prefix)) stays accurate when survival is near one.
        one_minus = -np.expm1(log_prefix)
        value = -np.mean(labels * log_prefix + (1 - labels) * np.log(np.maximum(one_minus, 1e-300)))
        derivative = (prefix - labels) / np.maximum(one_minus, 1e-300) / prefix.size
    else:
        raise ValueError("joint loss must be brier or bce")
    complement = sigmoid(-(logits * scale + offset))
    dz = np.cumsum(derivative[:, ::-1], axis=1)[:, ::-1] * complement
    gradient = np.r_[(dz[:, active] * logits[:, active] * alpha).sum(), dz[:, active].sum(axis=0)]
    value += regularization * np.mean(parameters**2)
    gradient += 2 * regularization * parameters / len(parameters)
    return float(value), gradient


def fit_joint(raw, accepted, min_observed=20, loss="brier", regularization=1e-4):
    from scipy.optimize import minimize

    labels, observed = outcomes(accepted, raw.shape[1])
    active = observed.sum(axis=0) >= min_observed
    if not active.any():
        return {"kind": "identity", "gamma": raw.shape[1], "fit": {"active_depths": [], "reason": "insufficient observations"}}
    p = np.clip(raw, EPS, 1 - EPS)
    logits = np.log(p) - np.log1p(-p)
    result = minimize(joint_objective, np.zeros(1 + int(active.sum())), args=(logits, labels, active, loss, regularization), jac=True, method="L-BFGS-B", bounds=[(math.log(0.05), math.log(20))] + [(-12, 12)] * int(active.sum()), options={"maxiter": 500, "ftol": 1e-12})
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise RuntimeError("joint calibration optimization was non-finite")
    if not result.success:
        raise RuntimeError(f"joint calibration did not converge: {result.message}")
    scale = np.where(active, math.exp(result.x[0]), 1.0)
    offset = np.zeros(raw.shape[1])
    offset[active] = result.x[1:]
    return {"kind": "logit", "gamma": raw.shape[1], "scale": scale.tolist(), "offset": offset.tolist(), "eps": EPS, "fit": {"shared_alpha": float(math.exp(result.x[0])), "active_depths": np.flatnonzero(active).tolist(), "loss": loss, "regularization": regularization, "converged": bool(result.success), "objective": float(result.fun)}}


def lookup_model(model, bins, grid="linear"):
    if bins < 2:
        raise ValueError("lookup bins must be >=2")
    boundaries = None
    if grid == "linear":
        representatives = (np.arange(bins) + 0.5) / bins
    elif grid == "logit":
        limit = math.log((1 - model.get("eps", EPS)) / model.get("eps", EPS))
        logits = np.linspace(-limit, limit, bins)
        representatives = sigmoid(logits)
        representatives[0], representatives[-1] = 0.0, 1.0
        boundaries = sigmoid((logits[1:] + logits[:-1]) / 2)
    else:
        raise ValueError("lookup grid must be linear or logit")
    raw = np.broadcast_to(representatives[:, None], (bins, model["gamma"]))
    table = predict(model, raw).T
    result = {"kind": "table", "gamma": model["gamma"], "probabilities": table.tolist(), "monotonic": bool(np.all(np.diff(table, axis=1) >= 0)), "lookup_grid": grid}
    if boundaries is not None:
        result.update(boundaries=boundaries.tolist(), representatives=representatives.tolist(), eps=model.get("eps", EPS))
    return result


def fit_family(calib, val, test, bins=16, lookup_bins=(16, 256, 1024), min_observed=20, ece_bins=15, joint_loss="brier", legacy=None, evaluate_test=False):
    sources = {confidence_source(row) for row in calib + val}
    if len(sources) != 1:
        raise ValueError("one mode mixes different confidence sources across calibration/validation")
    source = next(iter(sources))
    if evaluate_test and any(confidence_source(row) != source for row in test):
        raise ValueError("test confidence source differs from the frozen calibration feature")
    raw, accepted = arrays(calib)
    gamma = raw.shape[1]
    success, count = bin_counts(raw, accepted, bins)
    candidates = {
        "identity": {"kind": "identity", "gamma": gamma},
        f"empirical_{bins}": legacy_model({"success": success, "count": count}),
        f"isotonic_{bins}": fit_isotonic(raw, accepted, bins, min_observed),
        "sts": fit_sts(raw, accepted, min_observed, ece_bins),
        "joint_logit": fit_joint(raw, accepted, min_observed, joint_loss),
    }
    if legacy is not None:
        candidates["legacy_map"] = legacy_model(legacy)
        if candidates["legacy_map"]["gamma"] != gamma:
            raise ValueError("legacy calibration gamma differs from input")
    val_raw, val_accepted = arrays(val)
    val_legacy = legacy_features(val) if legacy is not None else None
    val_metrics = {name: evaluate(model, val_legacy if name == "legacy_map" else val_raw, val_accepted, ece_bins) for name, model in candidates.items()}
    # The predeclared selector is prefix Brier on val. No test metric enters it.
    eligible = [name for name, model in candidates.items() if model["kind"] != "table" or model.get("monotonic", False)]
    winner = min(eligible, key=lambda name: val_metrics[name]["prefix"]["brier"])
    selected = candidates[winner]
    tables = {str(n): lookup_model(selected, n) for n in lookup_bins}
    tables.update({f"logit{n}": lookup_model(selected, n, "logit") for n in lookup_bins})
    report = {"selected": winner, "selection": "minimum validation micro-prefix Brier among order-preserving candidates; calibration-only fitting", "selection_candidates": eligible, "validation": val_metrics, "test": {}, "test_scope": "withheld" if not evaluate_test else "frozen winner and legacy only", "quantization": {}, "selected_by_group": {}}
    if legacy is not None:
        report["legacy_map_feature"] = "greedy_confidence; confidence only for old greedy-proxy records"
    splits = {"val": (val_raw, val_accepted)}
    if test and evaluate_test:
        test_arrays = arrays(test)
        test_models = {winner: selected}
        if "legacy_map" in candidates:
            test_models["legacy_map"] = candidates["legacy_map"]
        report["test"] = {name: evaluate(model, legacy_features(test) if name == "legacy_map" else test_arrays[0], test_arrays[1], ece_bins) for name, model in test_models.items()}
    for split, (x, a) in splits.items():
        exact = predict(selected, x)
        exact_prefix = np.cumprod(exact, axis=1)
        report["quantization"][split] = {}
        for n, table in tables.items():
            quantized = predict(table, x)
            report["quantization"][split][n] = {"conditional_mae": float(np.abs(quantized - exact).mean()), "conditional_max_error": float(np.abs(quantized - exact).max()), "prefix_mae": float(np.abs(np.cumprod(quantized, axis=1) - exact_prefix).mean()), "metrics": evaluate(table, x, a, ece_bins)}
        source_rows = val if split == "val" else test
        report["selected_by_group"][split] = {}
        for field in ("benchmark", "reasoning", "batch_size", "sampling_params"):
            groups = {}
            for row in source_rows:
                if field in row:
                    key = json.dumps(row[field], sort_keys=True)
                    groups.setdefault(key, []).append(row)
            if groups:
                report["selected_by_group"][split][field] = {key: evaluate(selected, *arrays(group), ece_bins) for key, group in sorted(groups.items())}
    _, observed = outcomes(accepted, gamma)
    evidence = {
        "split_rounds": {"calib": len(calib), "val": len(val), "test": len(test)},
        "observed_conditional_by_depth": observed.sum(axis=0).tolist(),
        "calibration_confidence_bin_count": count.tolist(),
        "sampling_params": sorted({json.dumps(r.get("sampling_params", {}), sort_keys=True) for r in calib + val}),
        "confidence_source": source,
        "scope": "The source distinguishes pre-token maximum temperature-softmax probability before truncation, filtered-q maximum on the actual sampled chain, and legacy greedy-chain proxy features. Fitting uses confidence only; sampling parameters describe evidence and do not gate runtime.",
    }
    return {"selected": winner, "model": selected, "lookup_tables": tables, "confidence_source": source, "evidence": evidence}, report


def run(rows, runtime_lookup_bins=1024, lookup_grid="logit", **kwargs):
    if runtime_lookup_bins not in kwargs.get("lookup_bins", (16, 256, 1024)):
        raise ValueError("runtime lookup size must occur in lookup_bins")
    artifact = {"format": FORMAT, "gamma": len(rows[0]["confidence"]), "lookup_bins": runtime_lookup_bins, "lookup_grid": lookup_grid, "label_scope": "full-block accepted speculative prefix excluding anchor and bonus", "families": {}}
    report = {"format": FORMAT, "note": "No result is evidence of serving throughput; selection uses validation only. Conditional labels after first failure are censored; prefix outcomes are fully observed.", "families": {}}
    for mode in sorted({r["mode"] for r in rows}):
        splits = {s: [r for r in rows if r["mode"] == mode and r["split"] == s] for s in ("calib", "val", "test")}
        if not splits["calib"] or not splits["val"]:
            raise ValueError(f"{mode} needs nonempty conversation-disjoint calib and val")
        artifact["families"][mode], report["families"][mode] = fit_family(**splits, **kwargs)
    artifact["confidence_sources"] = {mode: family["confidence_source"] for mode, family in artifact["families"].items()}
    sources = set(artifact["confidence_sources"].values())
    artifact["confidence_source"] = next(iter(sources)) if len(sources) == 1 else "per_mode"
    return artifact, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--legacy-calibration", type=Path)
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--lookup-bins", type=int, nargs="+", default=[16, 256, 1024])
    parser.add_argument("--runtime-lookup-bins", type=int, default=1024)
    parser.add_argument("--lookup-grid", choices=["linear", "logit"], default="logit")
    parser.add_argument("--min-observed", type=int, default=20)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--joint-loss", choices=["brier", "bce"], default="brier")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evaluate-test", action="store_true", help="final evaluation only: report winner/legacy after all model, lookup and budget choices are frozen")
    parser.add_argument("--no-test", dest="evaluate_test", action="store_false", help="withhold all test metrics (the default)")
    parser.set_defaults(evaluate_test=False)
    args = parser.parse_args()
    if args.bins < 2 or args.ece_bins < 2 or args.min_observed < 1 or min(args.lookup_bins) < 2:
        parser.error("bins must be >=2 and min-observed >=1")
    if args.runtime_lookup_bins not in args.lookup_bins:
        parser.error("--runtime-lookup-bins must occur in --lookup-bins")
    rows = load_rounds(args.input, args.seed)
    legacy = json.loads(args.legacy_calibration.read_text()) if args.legacy_calibration else None
    artifact, report = run(rows, runtime_lookup_bins=args.runtime_lookup_bins, lookup_grid=args.lookup_grid, bins=args.bins, lookup_bins=args.lookup_bins, min_observed=args.min_observed, ece_bins=args.ece_bins, joint_loss=args.joint_loss, legacy=legacy, evaluate_test=args.evaluate_test)
    artifact["split_seed"] = args.seed
    for path, value in ((args.output, artifact), (args.report, report)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps({mode: family["selected"] for mode, family in artifact["families"].items()}))


if __name__ == "__main__":
    main()
