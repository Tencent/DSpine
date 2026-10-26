"""Batched refinement-tail helpers for the SGLang MLX worker.

The single-stream ``dspine_mlx.propose.RefinementTail`` stays untouched.  These
helpers reuse its vocabulary tables, head accessor and constants, and only add a
batch dimension: rows are request-major, and each request's predecessor chain
never crosses a request boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from .kernels import dispatch


@dataclass
class BatchedTailOutput:
    tokens: mx.array          # [B, T-1] int32
    candidates: mx.array      # [B, T-1, K] int32
    q_rows: mx.array | None   # [B, T-1, K] fp32 for sampled requests


def _split_stash(tail, stash: dict, batch: int, block: int):
    T = int(block)
    raw = stash["raw"]
    rms = stash["rms"]
    packed = stash["packed"]
    if raw.shape[0] != batch * T:
        raise ValueError(
            f"batched stash has {raw.shape[0]} rows, expected {batch * T}"
        )
    return (
        raw.reshape(batch, T, -1)[:, 1:].astype(mx.float32),
        rms.reshape(batch, T)[:, 1:].astype(mx.float32),
        packed.reshape(batch, T, -1)[:, 1:, : tail.rank].astype(mx.float32),
    )


def sid_batched(cand: mx.array, anchors: mx.array) -> mx.array:
    """Predecessor ids per request; never crosses a request boundary."""
    batch, _n, k = cand.shape
    anchor = mx.broadcast_to(
        anchors.astype(mx.int32)[:, None, None], (batch, 1, k)
    )
    return mx.concatenate([anchor, cand[:, :-1]], axis=1)


def _prepare(tail, rows, stash, anchors, batch: int, block: int):
    n = int(block) - 1
    if rows.shape[0] != batch * n:
        raise ValueError(
            f"batched tail got {rows.shape[0]} rows, expected {batch * n}"
        )
    raw, rms, q = _split_stash(tail, stash, batch, block)
    cand = tail.candidates(rows).reshape(batch, n, tail.k)
    return cand, raw, rms, q, sid_batched(cand, anchors)


def _greedy_scores(tail, cand, raw, rms, q, sid):
    head_rows = tail.head.rows(cand).astype(mx.float32)  # [B,n,K,H]
    base = mx.sum((raw[:, :, None, :] * tail.norm_w) * head_rows, axis=-1)
    tsv_sid = tail.tsv[sid].astype(mx.float32)  # [B,n,K,2R]
    tvg = tsv_sid[..., : tail.rank]
    tvv = tsv_sid[..., tail.rank :]
    m = mx.sigmoid(q[:, :, None, :] + tvg + tail.gate_b) * tvv
    tg_cand = tail.tg[cand].astype(mx.float32)
    s = mx.matmul(m, tg_cand.swapaxes(-1, -2))
    score = base[:, :, None, :] + rms[:, :, None, None] * s
    return mx.argmax(score, axis=-1).astype(mx.int32)  # [B,n,K]


def _sample_table(tail, cand, raw, rms, q, sid):
    e = tail.sqrt_c * tail.codebook[sid].astype(mx.float32)  # [B,n,K,C]
    gate = mx.sigmoid(
        q[:, :, None, :] + mx.matmul(e, tail.pred_gate_w.T) + tail.gate_b
    )
    value = mx.matmul(e, tail.prev_w.T)
    refined = raw[:, :, None, :] + rms[:, :, None, None] * mx.matmul(
        gate * value, tail.write_w.T
    )
    norm = mx.sqrt(mx.mean(refined * refined, axis=-1, keepdims=True) + tail.eps)
    normalized = refined / norm * tail.norm_w
    head_rows = tail.head.rows(cand).astype(mx.float32)
    return mx.matmul(normalized, head_rows.swapaxes(-1, -2))  # [B,n,K,K]


def greedy_batched(tail, rows, stash, anchors, *, batch: int, block: int):
    cand, raw, rms, q, sid = _prepare(tail, rows, stash, anchors, batch, block)
    idx = _greedy_scores(tail, cand, raw, rms, q, sid)
    n, k = int(block) - 1, tail.k
    tokens = dispatch.greedy_chain(
        idx.reshape(batch * n, k),
        cand.reshape(batch * n, k),
        block_size=int(block),
    ).reshape(batch, n)
    return BatchedTailOutput(tokens=tokens, candidates=cand, q_rows=None)


def sampled_table_batched(tail, rows, stash, anchors, *, batch: int, block: int):
    cand, raw, rms, q, sid = _prepare(tail, rows, stash, anchors, batch, block)
    return _sample_table(tail, cand, raw, rms, q, sid), cand


def sampled_batched(
    tail,
    rows,
    stash,
    anchors,
    *,
    inv_temp,
    top_k,
    top_p,
    min_p,
    uniforms,
    batch: int,
    block: int,
):
    table, cand = sampled_table_batched(
        tail, rows, stash, anchors, batch=batch, block=block
    )
    n, k = int(block) - 1, tail.k
    tokens, q_rows = dispatch.serial_sample_chain(
        table.reshape(batch * n, k, k),
        cand.reshape(batch * n, k),
        inv_temp=inv_temp,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        uniforms=uniforms,
        block_size=int(block),
    )
    return BatchedTailOutput(
        tokens=tokens.reshape(batch, n),
        candidates=cand,
        q_rows=q_rows.reshape(batch, n, k),
    )


def forward_batched(
    tail,
    rows,
    stash,
    anchors,
    *,
    greedy_mask: mx.array,
    inv_temp,
    top_k,
    top_p,
    min_p,
    uniforms,
    batch: int,
    block: int,
):
    """Mixed greedy/sampled batch: computes both branches and selects."""
    cand, raw, rms, q, sid = _prepare(tail, rows, stash, anchors, batch, block)
    n, k = int(block) - 1, tail.k
    idx = _greedy_scores(tail, cand, raw, rms, q, sid)
    greedy_tokens = dispatch.greedy_chain(
        idx.reshape(batch * n, k),
        cand.reshape(batch * n, k),
        block_size=int(block),
    ).reshape(batch, n)
    table = _sample_table(tail, cand, raw, rms, q, sid)
    sampled_tokens, q_rows = dispatch.serial_sample_chain(
        table.reshape(batch * n, k, k),
        cand.reshape(batch * n, k),
        inv_temp=inv_temp,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        uniforms=uniforms,
        block_size=int(block),
    )
    tokens = mx.where(
        greedy_mask[:, None], greedy_tokens, sampled_tokens.reshape(batch, n)
    )
    return BatchedTailOutput(
        tokens=tokens,
        candidates=cand,
        q_rows=q_rows.reshape(batch, n, k),
    )
