from __future__ import annotations

import mlx.core as mx

from .propose import filter_target_probs


def _leading_true(mask: mx.array) -> mx.array:
    if mask.size == 0:
        return mx.array(0, dtype=mx.int32)
    return mx.sum(mx.cumprod(mask.astype(mx.int32), axis=0)).astype(mx.int32)


def greedy_verify(logits: mx.array, draft_tokens: mx.array):
    """Exact-argmax verification of a block. Returns (accept_len, bonus)."""
    target = mx.argmax(logits, axis=-1).astype(mx.int32)
    matches = draft_tokens == target[:-1]
    accept_len = _leading_true(matches)
    bonus = mx.take(target, accept_len)
    return accept_len, bonus


def sample_from_probs(probs: mx.array, u: mx.array) -> mx.array:
    cdf = mx.cumsum(probs, axis=-1)
    threshold = u * cdf[-1]
    choice = mx.sum((cdf <= threshold).astype(mx.int32))
    return mx.minimum(choice, probs.shape[-1] - 1).astype(mx.int32)


def sampled_verify(
    logits: mx.array,
    draft_tokens: mx.array,
    cand: mx.array,
    q_rows: mx.array,
    *,
    inv_temp: float,
    top_k: int,
    top_p: float,
    min_p: float,
    u_accept: mx.array,
    u_final: mx.array,
    u_res: mx.array,
):
    """Chain rejection sampling against the target distribution.

    Returns (accept_len, bonus). Committed tokens follow the target's filtered
    distribution for any draft q.
    """
    probs = mx.softmax(
        logits.astype(mx.float32) * float(inv_temp), axis=-1
    )
    p = filter_target_probs(probs, top_k, top_p, min_p)  # [T, V]
    n = draft_tokens.shape[0]
    p_d = mx.take_along_axis(p[:n], draft_tokens[:, None], axis=-1)[:, 0]
    q_d = mx.sum(
        mx.where(cand == draft_tokens[:, None], q_rows, 0.0), axis=-1
    )
    ratio = p_d / mx.maximum(q_d, 1e-30)
    accept = u_accept < mx.minimum(1.0, ratio)
    accept_len = _leading_true(accept)

    q_dense = mx.put_along_axis(
        mx.zeros((n, p.shape[1]), dtype=mx.float32), cand, q_rows, axis=1
    )
    res_row = mx.minimum(accept_len, n - 1)
    p_res = mx.take(p[:n], res_row, axis=0)
    q_res = mx.take(q_dense, res_row, axis=0)
    residual = mx.maximum(p_res - q_res, 0.0)
    residual = residual / mx.sum(residual)
    bonus_res = sample_from_probs(residual, u_res)
    bonus_final = sample_from_probs(p[n], u_final)
    bonus = mx.where(accept_len == n, bonus_final, bonus_res).astype(mx.int32)
    return accept_len, bonus
