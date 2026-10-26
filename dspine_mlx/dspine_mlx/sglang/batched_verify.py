"""Batched greedy/sampled verification for the SGLang MLX worker.

The single-request functions remain in ``dspine_mlx.sampler``; these variants
remove the per-request device-to-host transfers and support per-request
sampling parameters.
"""

from __future__ import annotations

import mlx.core as mx

from .kernels.filter import normalize_rows


def _leading_true_rows(mask: mx.array) -> mx.array:
    """Length of the leading True run on the last axis of ``[B, n]``."""
    if mask.size == 0:
        return mx.zeros((mask.shape[0],), dtype=mx.int32)
    return mx.sum(mx.cumprod(mask.astype(mx.int32), axis=-1), axis=-1).astype(
        mx.int32
    )


def greedy_verify_batched(
    logits: mx.array, draft_tokens: mx.array
) -> tuple[mx.array, mx.array]:
    """Batched exact-argmax verification. Returns ``(accept [B], bonus [B])``."""
    target = mx.argmax(logits, axis=-1).astype(mx.int32)  # [B, T]
    n = draft_tokens.shape[-1]
    matches = draft_tokens == target[:, :n]
    accept = _leading_true_rows(matches)
    bonus = mx.take_along_axis(target, accept[:, None], axis=1)[:, 0]
    return accept, bonus


def _sample_rows(probs: mx.array, u: mx.array) -> mx.array:
    """Inverse-CDF sample on the last axis of ``[B, V]`` with ``u [B]``."""
    cdf = mx.cumsum(probs, axis=-1)
    threshold = u * cdf[:, -1]
    choice = mx.sum((cdf <= threshold[:, None]).astype(mx.int32), axis=-1)
    return mx.minimum(choice, probs.shape[-1] - 1).astype(mx.int32)


def filter_target_probs_batched(
    probs: mx.array,
    top_k: mx.array,
    top_p: mx.array,
    min_p: mx.array,
    *,
    skip_noop: bool = False,
    apply_min_p: bool = False,
) -> mx.array:
    """Joint top-k / top-p + min-p over ``[B, T, V]`` with per-request params."""
    if skip_noop:
        return normalize_rows(probs)
    order = mx.argsort(-probs, axis=-1)
    rank = mx.argsort(order, axis=-1)
    sorted_p = mx.take_along_axis(probs, order, axis=-1)
    cum = mx.cumsum(sorted_p, axis=-1)
    before = mx.take_along_axis(cum - sorted_p, rank, axis=-1)
    keep = (rank < top_k[:, None, None]) & (before <= top_p[:, None, None])
    if apply_min_p:
        pmax = mx.max(probs, axis=-1, keepdims=True)
        keep = keep & (probs >= min_p[:, None, None] * pmax)
    w = mx.where(keep, probs, 0.0)
    return w / mx.sum(w, axis=-1, keepdims=True)


def sampled_verify_batched(
    logits: mx.array,
    draft_tokens: mx.array,
    cand: mx.array,
    q_rows: mx.array,
    *,
    inv_temp: mx.array,
    top_k: mx.array,
    top_p: mx.array,
    min_p: mx.array,
    u_accept: mx.array,
    u_final: mx.array,
    u_res: mx.array,
    skip_noop: bool = False,
    apply_min_p: bool = False,
) -> tuple[mx.array, mx.array]:
    """Batched chain rejection sampling. Returns ``(accept [B], bonus [B])``.

    ``logits`` is ``[B, T, V]``, ``draft_tokens``/``cand``/``q_rows`` are
    ``[B, T-1, ...]``, and the sampling parameters are ``[B]`` arrays.
    """
    batch, _T, V = logits.shape
    n = draft_tokens.shape[1]
    probs = mx.softmax(
        logits.astype(mx.float32) * inv_temp[:, None, None], axis=-1
    )
    p = filter_target_probs_batched(
        probs,
        top_k,
        top_p,
        min_p,
        skip_noop=skip_noop,
        apply_min_p=apply_min_p,
    )  # [B, T, V]
    p_d = mx.take_along_axis(p[:, :n], draft_tokens[:, :, None], axis=-1)[
        :, :, 0
    ]
    q_d = mx.sum(
        mx.where(cand == draft_tokens[:, :, None], q_rows, 0.0), axis=-1
    )
    ratio = p_d / mx.maximum(q_d, 1e-30)
    accept = u_accept < mx.minimum(1.0, ratio)
    accept_len = _leading_true_rows(accept)

    q_dense = mx.put_along_axis(
        mx.zeros((batch, n, V), dtype=mx.float32), cand, q_rows, axis=2
    )
    res_row = mx.minimum(accept_len, n - 1)
    p_res = mx.take_along_axis(
        p[:, :n], res_row[:, None, None], axis=1
    )[:, 0, :]
    q_res = mx.take_along_axis(
        q_dense, res_row[:, None, None], axis=1
    )[:, 0, :]
    residual = mx.maximum(p_res - q_res, 0.0)
    residual = residual / mx.sum(residual, axis=-1, keepdims=True)
    bonus_res = _sample_rows(residual, u_res)
    bonus_final = _sample_rows(p[:, n], u_final)
    bonus = mx.where(
        accept_len == n, bonus_final, bonus_res
    ).astype(mx.int32)
    return accept_len, bonus
