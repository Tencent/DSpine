"""Target-distribution filter helpers.

The reference implementation lives in :mod:`dspine_mlx.propose`; this module
only identifies the common "filters are a no-op" case so the caller can skip
two full-vocabulary argsorts.  Skipping renormalizes an already-normalized
distribution, which differs from the reference by ~1e-7 in fp32 (distribution-
identical, not bit-identical), so it is controlled by ``filter_fastpath``.
"""

from __future__ import annotations

import mlx.core as mx


def is_noop(top_k: int, top_p: float, min_p: float, vocab_size: int) -> bool:
    return (
        int(top_k) >= int(vocab_size)
        and float(top_p) >= 1.0
        and float(min_p) <= 0.0
    )


def normalize_rows(probs: mx.array) -> mx.array:
    return probs / mx.sum(probs, axis=-1, keepdims=True)
