"""Pure-MLX reference implementations of the serial chain kernels.

These are the originals from ``dspine_mlx.propose``; they stay as the fallback
for the MSL kernels and as the reference in parity tests.
"""

from __future__ import annotations

import mlx.core as mx


def chain(idx: mx.array, cand: mx.array) -> mx.array:
    """Walk the K-state greedy table; returns the selected candidate per row."""
    n = cand.shape[0]
    j = idx[0, 0]
    out = []
    for i in range(n):
        out.append(mx.take(cand[i], j))
        if i + 1 < n:
            j = mx.take(idx[i + 1], j)
    return mx.stack(out).astype(mx.int32)


def serial_sample_chain(
    table: mx.array,
    cand: mx.array,
    *,
    inv_temp: float,
    top_k: int,
    top_p: float,
    min_p: float,
    uniforms: mx.array,
) -> tuple[mx.array, mx.array]:
    """Sample each row's chain under the request's filters."""
    n, K, _ = table.shape
    lane = mx.arange(K)
    sender = mx.array(0, dtype=mx.int32)
    toks = []
    qs = []
    for i in range(n):
        row = mx.take(table[i], sender, axis=0)
        p = mx.softmax(row * float(inv_temp), axis=-1)
        q, keep = _filter_probs(p, top_k, top_p, min_p)
        cdf = mx.cumsum(q, axis=-1)
        threshold = uniforms[i] * cdf[-1]
        choice = mx.sum((cdf <= threshold).astype(mx.int32))
        last = mx.max(mx.where(keep, lane, 0))
        choice = mx.minimum(choice, last)
        toks.append(mx.take(cand[i], choice))
        qs.append(q)
        sender = choice
    return mx.stack(toks).astype(mx.int32), mx.stack(qs)


def _filter_probs(
    p: mx.array, top_k: int, top_p: float, min_p: float
) -> tuple[mx.array, mx.array]:
    K = p.shape[0]
    lane = mx.arange(K)
    p_c = p[:, None]
    p_d = p[None, :]
    ahead = (p_d > p_c) | ((p_d == p_c) & (lane[None, :] < lane[:, None]))
    rank = mx.sum(ahead.astype(mx.int32), axis=1)
    before = mx.sum(mx.where(ahead, p_d, 0.0), axis=1)
    keep = (rank < int(top_k)) & (before <= float(top_p))
    if min_p > 0:
        keep = keep & (p >= float(min_p) * mx.max(p))
    w = mx.where(keep, p, 0.0)
    q = w / mx.sum(w)
    return q, keep
