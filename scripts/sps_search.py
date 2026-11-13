"""Task-separated, paired search of one actual-batch SPS entry at a time."""

import copy
import hashlib
import json
import math


def task_key(row):
    identity = next((row[k] for k in ("task_id", "conversation_id", "id") if row.get(k) is not None), None)
    if identity is None:
        identity = hashlib.sha256(json.dumps(row["input_ids"]).encode()).hexdigest()
    return json.dumps([row.get("dataset", row.get("source")), identity], sort_keys=True)


def split_tasks(prefixes, holdout=None):
    """Keep every turn and sampler variant of one task in the same partition."""
    if holdout is None:
        groups = {}
        for row in prefixes:
            groups.setdefault(task_key(row), []).append(row)
        keys = sorted(groups, key=lambda k: hashlib.sha256(("sps-holdout-v2:" + k).encode()).hexdigest())
        if len(keys) < 3:
            raise ValueError("Provide at least three independent tasks or an independent holdout file")
        held = set(keys[:max(1, len(keys) // 3)])
        # Repair the deterministic split if a sparse sampling family is absent.
        modes = {row["_mode"] for row in prefixes}
        for _ in range(4):
            for mode in sorted(modes):
                for want_holdout in (True, False):
                    target = [k for k in keys if (k in held) == want_holdout]
                    if any(row["_mode"] == mode for k in target for row in groups[k]):
                        continue
                    for key in keys:
                        if (key in held) == want_holdout or not any(r["_mode"] == mode for r in groups[key]):
                            continue
                        remaining = [k for k in keys if (k in held) != want_holdout and k != key]
                        if all(any(r["_mode"] == m for k in remaining for r in groups[k]) for m in modes):
                            held.symmetric_difference_update({key})
                            break
        tune = [r for k in keys if k not in held for r in groups[k]]
        holdout = [r for k in keys if k in held for r in groups[k]]
    else:
        tune = list(prefixes)
        holdout = list(holdout)
    tune_keys, holdout_keys = {task_key(r) for r in tune}, {task_key(r) for r in holdout}
    if tune_keys & holdout_keys:
        raise ValueError("Tuning and holdout task identities overlap")
    token_keys = lambda rows: {hashlib.sha256(json.dumps(r["input_ids"]).encode()).hexdigest() for r in rows}
    if token_keys(tune) & token_keys(holdout):
        raise ValueError("Tuning and holdout input tokens overlap")
    for mode in {r["_mode"] for r in tune + holdout}:
        if not any(r["_mode"] == mode for r in tune) or not any(r["_mode"] == mode for r in holdout):
            raise ValueError(f"Both tuning and holdout need independent {mode} tasks")
    if not tune or not holdout:
        raise ValueError("Tuning and holdout must both be nonempty")
    return tune, holdout, {"task_disjoint": True, "input_ids_disjoint": True,
        "tune_tasks": sorted(tune_keys), "holdout_tasks": sorted(holdout_keys),
        "tune_rows": len(tune), "holdout_rows": len(holdout)}


def replace_budget(document, mode, bs, budget):
    """Change one mode/actual-B entry while preserving the complete baseline."""
    if type(bs) is not int or type(budget) is not int or not bs <= budget <= bs * 16:
        raise ValueError("Invalid batch or token budget")
    result = copy.deepcopy(document)
    policies = result.setdefault("mode_policies", {})
    policy = policies.setdefault(mode, copy.deepcopy(result["policy"]))
    policy.setdefault("verify_widths", {}).pop(str(bs), None)
    policy.setdefault("token_budgets", {})[str(bs)] = budget
    return result


def current_budget(document, mode, bs):
    policy = document.get("mode_policies", {}).get(mode, document["policy"])
    return policy.get("token_budgets", {}).get(str(bs), bs * policy.get("verify_widths", {}).get(str(bs), 16))


def aggregate(rounds, *, min_seconds=0, min_requests=1):
    if not rounds:
        return {"status": "INSUFFICIENT", "rounds": []}
    adequate = all(r["status"] == "PASS" and r["wall_seconds"] >= min_seconds
                   and r["requests"] >= min_requests for r in rounds)
    wall = sum(r["wall_seconds"] for r in rounds)
    return {"status": "PASS" if adequate else "INSUFFICIENT", "rounds": rounds,
        "tokens_per_second": sum(r["output_tokens"] for r in rounds) / wall,
        "speculative_verification_observed": any(r["verify_calls"] for r in rounds)}


def paired_decision(reference, candidate, min_gain=0.03):
    """Require a material pooled gain and the same direction in every repeat."""
    before, after = reference["rounds"], candidate["rounds"]
    signature = lambda rows: [(r["repeat"], r["requests"], r["jobs_sha256"]) for r in rows]
    if signature(before) != signature(after):
        raise ValueError("Candidates must use paired complete requests, seeds and repeats")
    if len({r["repeat"] for r in before}) != len(before):
        raise ValueError("Duplicate repeat in a paired comparison")
    ratios = [b["tokens_per_second"] / a["tokens_per_second"] for a, b in zip(before, after)]
    ratio = candidate["tokens_per_second"] / reference["tokens_per_second"]
    if not all(math.isfinite(x) and x > 0 for x in ratios + [ratio]):
        raise ValueError("Paired throughputs must be positive and finite")
    enough = reference.get("status") == candidate.get("status") == "PASS" and len(ratios) >= 2
    return {"adequate": enough, "throughput_ratio": ratio, "repeat_ratios": ratios,
        "min_gain": min_gain, "accepted": enough and ratio >= 1 + min_gain and all(x > 1 for x in ratios),
        "nonregressing": enough and ratio >= 1 / (1 + min_gain) and all(x >= 1 / (1 + 2 * min_gain) for x in ratios),
        "statistical_rule": "paired repeats, minimum pooled gain and consistent direction; no confidence interval claimed"}


def prefer_budget(measurements, baseline_budget, min_gain=0.03):
    valid = [r for r in measurements if r.get("status") == "PASS" and r.get("speculative_verification_observed")]
    reference = next((r for r in valid if r["token_budget"] == baseline_budget), None)
    if reference is None:
        return {"selected_budget": baseline_budget, "reason": "insufficient_baseline", "comparisons": []}
    best_rate = max(r["tokens_per_second"] for r in valid)
    close = [r for r in valid if r["tokens_per_second"] >= best_rate / (1 + min_gain)]
    candidate = max(close, key=lambda r: r["token_budget"])
    comparison = paired_decision(reference, candidate, min_gain)
    selected = candidate["token_budget"]
    if selected == baseline_budget:
        reason = "baseline_retained"
    elif comparison["accepted"]:
        reason = "material_paired_gain"
    elif selected > baseline_budget and comparison["nonregressing"]:
        reason = "wider_near_tie"
    else:
        selected, reason = baseline_budget, "gain_not_established"
    return {"selected_budget": selected, "reason": reason, "comparisons": [comparison],
        "candidate_budget": candidate["token_budget"]}


def interleaved_order(values, repeat):
    values = list(values)
    if repeat % 2:
        values.reverse()
    offset = (repeat // 2) % len(values)
    return values[offset:] + values[:offset]


def run_search(args, baseline, tune, holdout, folder, run_trial, candidate_budgets, save):
    """The trial callback may reuse an idle service; search semantics stay fixed."""
    records, decisions, neighbors, validation = [], [], [], []
    selected = copy.deepcopy(baseline)
    available = {}
    planned_gates, completed_gates = [], set()

    def gate_key(gate):
        return (gate["phase"], gate["mode"], gate["batch_size"], gate.get("neighbor_budget"))

    def finish(accepted, stop_reason=None):
        adequate = all(m["status"] == "PASS" for record in records for m in record["measurements"].values())
        report = {"status": "PASS" if adequate else "INSUFFICIENT_EVIDENCE",
                  "selected_profile": "candidate" if accepted else "baseline",
                  "selection_status": "VALIDATED" if accepted else "BASELINE_RETAINED",
                  "decisions": decisions, "neighbor_validation": neighbors,
                  "table_validation": validation, "measurements": records,
                  "stop_reason": stop_reason,
                  "skipped_gates": [gate for gate in planned_gates if gate_key(gate) not in completed_gates],
                  "selection_rule": "paired complete queues, minimum gain, wider near ties; independent holdout gates",
                  "min_gain": args.min_gain, "min_seconds": args.min_seconds, "min_requests": args.min_requests,
                  "repeats": args.repeats, "max_new_tokens": args.max_new_tokens}
        save(folder / "RESULTS.json", report)
        return selected if accepted else copy.deepcopy(baseline), report

    def matrix(documents, rows, bs, destination):
        count = max(args.min_requests, len(rows), 8 * bs)
        latest = None
        for attempt in range(3):
            rounds = {key: [] for key in documents}
            for repeat in range(args.repeats):
                for key in interleaved_order(documents, repeat):
                    path = destination / f"attempt{attempt}" / f"r{repeat}" / str(key)
                    trial = run_trial(documents[key], rows, bs, repeat, count, path)
                    rounds[key].append(trial)
            latest = {key: {**aggregate(values, min_seconds=args.min_seconds, min_requests=args.min_requests),
                            "token_budget": key} for key, values in rounds.items()}
            receipt = {"attempt": attempt, "requests_per_repeat": count, "batch_size": bs,
                       "measurements": latest}
            save(destination / f"attempt{attempt}" / "SUMMARY.json", receipt)
            if all(value["status"] == "PASS" for value in latest.values()):
                return latest
            # Grow the same complete queue for every candidate, never just one.
            shortest = min(r["wall_seconds"] for values in rounds.values() for r in values)
            count = max(count * 2, math.ceil(count * args.min_seconds / max(shortest, 0.001) * 1.1))
        return latest

    for mode in ("greedy", "sampled"):
        rows = [r for r in tune if r["_mode"] == mode]
        if not rows:
            continue
        for bs in sorted(set(args.batch_sizes)):
            old = current_budget(selected, mode, bs)
            budgets = sorted(set([old, bs * 16, *(candidate_budgets([bs], width, args.token_grid)[bs]
                                                   for width in args.widths)]))
            available[mode, bs] = budgets
            documents = {budget: replace_budget(selected, mode, bs, budget) for budget in budgets}
            result = matrix(documents, rows, bs, folder / "tune" / mode / f"b{bs}")
            decision = prefer_budget(list(result.values()), old, args.min_gain)
            decision.update(mode=mode, batch_size=bs, baseline_budget=old)
            decisions.append(decision)
            records.append({"phase": "tune", "mode": mode, "batch_size": bs,
                            "baseline_budget": old, "measurements": result})
            selected = replace_budget(selected, mode, bs, decision["selected_budget"])
            save(folder / "CANDIDATE.json", selected)
            save(folder / "PROGRESS.json", {"status": "TUNING", "decisions": decisions})

    neighbor_plan = []
    for (mode, bs), budgets in available.items():
        chosen = current_budget(selected, mode, bs)
        lower = [b for b in budgets if b < chosen]
        higher = [b for b in budgets if b > chosen]
        adjacent = ([max(lower)] if lower else []) + ([min(higher)] if higher else [])
        neighbor_plan.append((mode, bs, chosen, adjacent))
        planned_gates.extend({"phase": "holdout-neighbors", "mode": mode,
                              "batch_size": bs, "neighbor_budget": b} for b in adjacent)
    planned_gates.extend({"phase": "holdout-table", "mode": mode, "batch_size": bs}
                         for mode, bs in available)
    changed = any(current_budget(selected, mode, bs) != current_budget(baseline, mode, bs)
                  for mode, bs in available)
    if not changed:
        return finish(False, "no_policy_change")

    # Holdout never selects a new winner. A rejected gate cannot be repaired
    # by later gates, so retain the baseline without further GPU trials.
    gates = []
    for mode, bs, chosen, adjacent in neighbor_plan:
        rows = [r for r in holdout if r["_mode"] == mode]
        documents = {b: replace_budget(selected, mode, bs, b) for b in [chosen, *adjacent]}
        result = matrix(documents, rows, bs, folder / "holdout-neighbors" / mode / f"b{bs}")
        reference = result[chosen]
        for budget in adjacent:
            comparison = paired_decision(reference, result[budget], args.min_gain)
            # A new narrow entry needs a material held-out win over its wider neighbor.
            narrowed = chosen < budget and chosen != current_budget(baseline, mode, bs)
            clear_narrow_gain = paired_decision(result[budget], reference, args.min_gain)["accepted"]
            passed = comparison["adequate"] and not comparison["accepted"] and (not narrowed or clear_narrow_gain)
            gates.append(passed)
            neighbors.append({"mode": mode, "batch_size": bs, "chosen_budget": chosen,
                              "neighbor_budget": budget, "pass": passed, "comparison": comparison,
                              "narrower_choice_has_material_gain": clear_narrow_gain if narrowed else None})
            completed_gates.add(("holdout-neighbors", mode, bs, budget))
        records.append({"phase": "holdout-neighbors", "mode": mode, "batch_size": bs,
                        "measurements": result})
        if not all(row["pass"] for row in neighbors if row["mode"] == mode and row["batch_size"] == bs):
            return finish(False, "holdout_neighbor_gate_failed")

    for mode in ("greedy", "sampled"):
        rows = [r for r in holdout if r["_mode"] == mode]
        if not rows:
            continue
        for bs in sorted(set(args.batch_sizes)):
            result = matrix({"baseline": baseline, "candidate": selected}, rows, bs,
                            folder / "holdout-table" / mode / f"b{bs}")
            comparison = paired_decision(result["baseline"], result["candidate"], args.min_gain)
            passed = comparison["nonregressing"]
            gates.append(passed)
            validation.append({"mode": mode, "batch_size": bs, "pass": passed, **comparison})
            completed_gates.add(("holdout-table", mode, bs, None))
            records.append({"phase": "holdout-table", "mode": mode, "batch_size": bs,
                            "measurements": result})
            if not passed:
                return finish(False, "holdout_table_gate_failed")
    accepted = bool(gates) and all(gates) and changed
    return finish(accepted)
