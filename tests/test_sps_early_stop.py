"""A rejected holdout gate must not spend more trials on that candidate."""

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


class HoldoutEarlyStopTest(unittest.TestCase):
    def test_unchanged_policy_skips_all_holdout_trials(self):
        path = Path(__file__).resolve().parents[1] / "scripts/sps_search.py"
        spec = importlib.util.spec_from_file_location("sps_no_policy_change", path)
        search = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(search)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            baseline = {"policy": {"verify_widths": {}, "token_budgets": {}},
                        "mode_policies": {}}
            args = SimpleNamespace(batch_sizes=[8], widths=[4, 8, 12, 16], token_grid=[],
                                   repeats=3, min_requests=128, min_seconds=30,
                                   min_gain=.03, max_new_tokens=4096)
            rows = [{"_mode": mode} for mode in ("greedy", "sampled")]
            calls = []

            def trial(document, prefixes, bs, repeat, count, destination):
                self.assertIn("tune", destination.parts)
                calls.append(destination)
                return {"status": "PASS", "repeat": repeat, "requests": count,
                        "jobs_sha256": f"paired-{repeat}", "wall_seconds": 60,
                        "output_tokens": 6000, "tokens_per_second": 100,
                        "verify_calls": 100}

            def save(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))

            selected, report = search.run_search(args, baseline, rows, rows, folder, trial,
                lambda batches, width, grid: {bs: bs * width for bs in batches}, save)
            self.assertEqual(selected, baseline)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["selection_status"], "BASELINE_RETAINED")
            self.assertEqual(report["stop_reason"], "no_policy_change")
            self.assertEqual(len(calls), 24)
            self.assertEqual(len(report["measurements"]), 2)
            self.assertTrue(all(row["phase"] == "tune" for row in report["measurements"]))
            self.assertTrue(report["skipped_gates"])
            self.assertEqual(report["neighbor_validation"], [])
            self.assertEqual(report["table_validation"], [])
            self.assertTrue((folder / "RESULTS.json").exists())

    def test_rejected_gate_retains_records_and_skips_later_trials(self):
        path = Path(__file__).resolve().parents[1] / "scripts/sps_search.py"
        spec = importlib.util.spec_from_file_location("sps_early_stop", path)
        search = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(search)
        for rejected_phase in ("holdout-neighbors", "holdout-table"):
            with self.subTest(rejected_phase=rejected_phase), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                baseline = {"policy": {"verify_widths": {}, "token_budgets": {}},
                            "mode_policies": {mode: {"verify_widths": {}, "token_budgets": {"8": budget}}
                                              for mode, budget in (("greedy", 32), ("sampled", 64))}}
                args = SimpleNamespace(batch_sizes=[8], widths=[4, 8, 12, 16], token_grid=[],
                                       repeats=3, min_requests=128, min_seconds=30,
                                       min_gain=.03, max_new_tokens=4096)
                rows = [{"_mode": mode} for mode in ("greedy", "sampled")]
                calls = []

                def trial(document, prefixes, bs, repeat, count, destination):
                    mode = prefixes[0]["_mode"]
                    budget = search.current_budget(document, mode, bs)
                    phase = next(p for p in destination.parts
                                 if p in ("tune", "holdout-neighbors", "holdout-table"))
                    calls.append((phase, mode, repeat, budget))
                    rate = {32: 100, 64: 120, 96: 80, 128: 70}[budget]
                    if phase == rejected_phase and mode == "greedy":
                        rate = 80 if budget == 64 else 110
                    return {"status": "PASS", "repeat": repeat, "requests": count,
                            "jobs_sha256": f"paired-{mode}-{repeat}", "wall_seconds": 60,
                            "output_tokens": rate * 60, "tokens_per_second": rate,
                            "verify_calls": 100}

                def save(path, value):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(value))

                selected, report = search.run_search(args, baseline, rows, rows, folder, trial,
                    lambda batches, width, grid: {bs: bs * width for bs in batches}, save)
                self.assertEqual(selected, baseline)
                self.assertIsNot(selected, baseline)
                self.assertEqual(report["status"], "PASS")
                self.assertEqual(report["selection_status"], "BASELINE_RETAINED")
                self.assertEqual(report["stop_reason"], "holdout_neighbor_gate_failed"
                                 if rejected_phase == "holdout-neighbors" else "holdout_table_gate_failed")
                self.assertTrue(report["skipped_gates"])
                self.assertEqual(calls[-1][:2], (rejected_phase, "greedy"))
                self.assertFalse(any(phase == rejected_phase and mode == "sampled"
                                     for phase, mode, _, _ in calls))
                if rejected_phase == "holdout-neighbors":
                    self.assertFalse(any(phase == "holdout-table" for phase, _, _, _ in calls))
                    self.assertEqual(len(report["measurements"]), 3)
                else:
                    self.assertEqual(len(report["measurements"]), 5)
                self.assertEqual(json.loads((folder / "RESULTS.json").read_text()),
                                 json.loads(json.dumps(report)))
                self.assertTrue(list(folder.rglob("SUMMARY.json")))


if __name__ == "__main__":
    unittest.main()
