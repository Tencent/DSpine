"""Kernel configuration for the MLX DSpine implementation.

All flags are read once at import time (and can be overridden programmatically
with :func:`configure`).  Every optimization is opt-out so a deployment can fall
back to the pure-MLX reference path without changing code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

_DEFAULT_KERNELS = "chain,filter_fastpath"


@dataclass(frozen=True)
class KernelConfig:
    # MSL serial chain (greedy + sampled). Bitwise tokens; q within ~1e-7.
    chain: bool = True
    # MSL code bridge. Measured 1.14x on the bridge alone, no E2E gain; off.
    bridge: bool = False
    # Skip the full-vocab argsort when the target filters are a no-op.
    filter_fastpath: bool = True
    # Batch the refinement tail across requests (bit-exact on an 8-bit head).
    tail_batch: bool = True
    # Batch the draft backbone across requests (ULP-level proposal drift).
    draft_batch: bool = True
    # Compute/validate the reference implementation alongside MSL (tests).
    check: bool = False
    # Disable every optimization that is not bit-exact.
    strict: bool = False

    def without_inexact(self) -> "KernelConfig":
        """Return a config that only uses bit-exact paths."""
        return replace(
            self,
            chain=False,
            filter_fastpath=False,
            tail_batch=False,
            draft_batch=False,
            strict=True,
        )


def _flag(name: str) -> bool:
    return os.environ.get(name, "") in ("1", "true", "True", "yes", "on")


def _kernel_set() -> set[str]:
    raw = os.environ.get("DSPINE_MLX_KERNELS")
    if raw is None:
        # Legacy experiment flags.
        legacy = {
            x.strip().lower()
            for x in os.environ.get("DSPINE_METAL", "").split(",")
            if x.strip()
        }
        if legacy:
            if "all" in legacy:
                return {"chain", "bridge", "filter_fastpath"}
            names = set(legacy)
            if _flag("DSPINE_FILTER_FASTPATH"):
                names.add("filter_fastpath")
            return names
        raw = _DEFAULT_KERNELS
    names = {x.strip().lower() for x in raw.split(",") if x.strip()}
    if "none" in names:
        return set()
    if "all" in names:
        return {"chain", "bridge", "filter_fastpath"}
    return names


def from_env() -> KernelConfig:
    if _flag("DSPINE_MLX_NO_KERNELS"):
        return KernelConfig(
            chain=False,
            bridge=False,
            filter_fastpath=False,
            tail_batch=False,
            draft_batch=False,
            check=False,
            strict=True,
        )
    names = _kernel_set()
    cfg = KernelConfig(
        chain="chain" in names,
        bridge="bridge" in names,
        filter_fastpath="filter_fastpath" in names,
        check=_flag("DSPINE_MLX_KERNEL_CHECK"),
    )
    if _flag("DSPINE_MLX_STRICT_PARITY"):
        return cfg.without_inexact()
    if _flag("DSPINE_MLX_NO_BATCH_TAIL"):
        cfg = replace(cfg, tail_batch=False)
    if _flag("DSPINE_MLX_NO_BATCH_BACKBONE"):
        cfg = replace(cfg, draft_batch=False)
    return cfg


def batch_limits() -> tuple[int, int]:
    """``(max_prefix, max_spread)`` for the batched draft backbone."""
    def _int(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, default))
        except (TypeError, ValueError):
            return default

    return (
        _int("DSPINE_MLX_BATCH_MAX_PREFIX", 2048),
        _int("DSPINE_MLX_BATCH_MAX_SPREAD", 1024),
    )
