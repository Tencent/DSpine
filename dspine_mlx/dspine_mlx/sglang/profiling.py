"""Env-gated phase profiler for the DSpine MLX worker.

Profiling is off unless ``DSPINE_MLX_PROFILE=1``; when off the worker holds
``None`` and pays one branch per phase.  When on, phases accumulate host-side
wall time.  ``DSPINE_MLX_PROFILE_SYNC=1`` adds a ``mx.synchronize()`` at every
phase boundary so the numbers attribute device time; that serializes the round,
so throughput must be measured with profiling off.

A summary is written to the server log every ``DSPINE_MLX_PROFILE_EVERY``
rounds (default 100) as one ``DSPINE_MLX_PROFILE <json>`` line.  No HTTP
endpoint is added, so SGLang itself is not modified.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import time
from contextlib import contextmanager
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

LOG_PREFIX = "DSPINE_MLX_PROFILE"


class PhaseProfiler:
    """Per-phase timings and counters over a bounded window of rounds."""

    def __init__(
        self,
        *,
        sync: bool = False,
        every: int = 100,
        window: int = 256,
    ) -> None:
        self.sync = bool(sync)
        self.every = max(1, int(every))
        self.window = max(1, int(window))
        self._samples: dict[str, list[float]] = {}
        self._round_acc: dict[str, float] = {}
        self._rounds_by_batch: dict[int, list[dict[str, float]]] = {}
        self._counters: dict[str, int] = {}
        self._rounds = 0
        self._fields: dict[str, Any] = {}

    # ------------------------------------------------------------- recording
    @contextmanager
    def phase(self, name: str):
        if self.sync:
            mx.synchronize()
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            if self.sync:
                mx.synchronize()
            ms = (time.perf_counter_ns() - t0) / 1e6
            self._add(name, ms)
            self._round_acc[name] = self._round_acc.get(name, 0.0) + ms

    def mark(self, name: str, milliseconds: float) -> None:
        self._add(name, float(milliseconds))
        self._round_acc[name] = self._round_acc.get(name, 0.0) + float(milliseconds)

    def _add(self, name: str, value: float) -> None:
        samples = self._samples.get(name)
        if samples is None:
            samples = self._samples[name] = []
        samples.append(value)
        if len(samples) > self.window:
            del samples[: len(samples) - self.window]

    def count(self, name: str, value: int = 1) -> None:
        self._counters[name] = self._counters.get(name, 0) + int(value)

    def round_done(self, **fields: Any) -> None:
        """Close one decode round and periodically emit a summary line."""
        self._rounds += 1
        self._fields = dict(fields)
        requests = int(fields.get("requests", 0) or 0)
        bucket = self._rounds_by_batch.setdefault(requests, [])
        bucket.append(dict(self._round_acc))
        if len(bucket) > self.window:
            del bucket[: len(bucket) - self.window]
        self._round_acc = {}
        if self._rounds % self.every == 0:
            self.report()

    # -------------------------------------------------------------- reporting
    def snapshot(self) -> dict[str, Any]:
        phases = {}
        for name, samples in sorted(self._samples.items()):
            if not samples:
                continue
            ordered = sorted(samples)
            phases[name] = {
                "count": len(samples),
                "mean_ms": statistics.fmean(samples),
                "median_ms": statistics.median(samples),
                "p90_ms": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
                "sum_ms": sum(samples),
            }
        total = sum(row["mean_ms"] for row in phases.values())
        for row in phases.values():
            row["share"] = (row["mean_ms"] / total) if total else 0.0

        by_batch = {}
        for requests, rounds in sorted(self._rounds_by_batch.items()):
            if not rounds:
                continue
            names = sorted({name for rnd in rounds for name in rnd})
            row = {"rounds": len(rounds), "phases": {}}
            for name in names:
                values = [rnd.get(name, 0.0) for rnd in rounds]
                row["phases"][name] = {
                    "mean_ms": statistics.fmean(values),
                    "median_ms": statistics.median(values),
                }
            total = sum(p["mean_ms"] for p in row["phases"].values())
            for phase in row["phases"].values():
                phase["share"] = (phase["mean_ms"] / total) if total else 0.0
            by_batch[requests] = row

        return {
            "rounds": self._rounds,
            "sync": self.sync,
            "phases": phases,
            "by_batch": by_batch,
            "counters": dict(sorted(self._counters.items())),
            "last_round": dict(self._fields),
        }

    def report(self) -> dict[str, Any]:
        snap = self.snapshot()
        logger.info("%s %s", LOG_PREFIX, json.dumps(snap, sort_keys=True))
        return snap

    def reset(self) -> None:
        self._samples.clear()
        self._round_acc.clear()
        self._rounds_by_batch.clear()
        self._counters.clear()
        self._rounds = 0
        self._fields = {}


def make_profiler_from_env() -> PhaseProfiler | None:
    """Build a profiler from ``DSPINE_MLX_PROFILE*`` env vars, or ``None``."""
    if os.environ.get("DSPINE_MLX_PROFILE", "") not in ("1", "true", "True"):
        return None
    every = _int_env("DSPINE_MLX_PROFILE_EVERY", 100)
    window = _int_env("DSPINE_MLX_PROFILE_WINDOW", 256)
    sync = os.environ.get("DSPINE_MLX_PROFILE_SYNC", "") in ("1", "true", "True")
    prof = PhaseProfiler(sync=sync, every=every, window=window)
    logger.info(
        "DSPINE_MLX_PROFILE enabled (sync=%s every=%d window=%d)",
        sync,
        every,
        window,
    )
    return prof


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
