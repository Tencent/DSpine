"""Batched multi-token target verification for the SGLang MLX backend.

The upstream MLX runner only batches *one* new token per request
(``MLXAttentionWrapper._batched_decode``).  DSpine verifies a whole block of
``T`` tokens per request in one target forward, so this module adds a second
dispatch path to the same attention wrapper.

``verify.py`` deliberately imports **no** SGLang modules so its math can be
unit-tested in an MLX-only environment.  The plugin installs
``around_attention_call`` as a hook on ``MLXAttentionWrapper.__call__``.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable

import mlx.core as mx

_verify_ctx = threading.local()

# A/B escape hatch: the pre-E1/E2 pad+concat path, kept for parity checks.
_LEGACY_VERIFY = os.environ.get(
    "DSPINE_MLX_NO_VERIFY_FASTPATH", ""
) in ("1", "true", "True")


def set_verify_context(ctx: "VerifyContext | None") -> None:
    _verify_ctx.ctx = ctx


def get_verify_context() -> "VerifyContext | None":
    return getattr(_verify_ctx, "ctx", None)


def clear_verify_context() -> None:
    _verify_ctx.ctx = None


@dataclass
class VerifyContext:
    """State shared by every attention layer during one verify forward.

    ``caches[i]`` is request ``i``'s model-shaped cache list (indexed by the
    upstream layer index), so a wrapper resolves its own slot with
    ``caches[i][wrapper._layer_idx]``.

    The padding metadata and the causal/valid mask depend only on
    ``prefix_lens`` and ``num_tokens``, so they are built once here and reused
    by all target layers instead of being rebuilt per layer.
    """

    prefix_lens: list[int]
    caches: list[list[Any]]
    num_tokens: int
    offsets: mx.array = field(init=False)
    max_len: int = field(init=False)
    attn_mask: mx.array = field(init=False)

    def __post_init__(self) -> None:
        if len(self.prefix_lens) != len(self.caches):
            raise ValueError("prefix_lens and caches batch sizes differ")
        self.offsets = mx.array(self.prefix_lens, dtype=mx.int32)
        self.max_len = max(self.prefix_lens) + int(self.num_tokens)
        idx = mx.arange(self.max_len)
        valid = idx[None, :] < (self.offsets[:, None] + int(self.num_tokens))
        causal = idx[None, None, :] <= (
            self.offsets[:, None, None]
            + mx.arange(int(self.num_tokens))[None, :, None]
        )
        self.attn_mask = (valid[:, None, :] & causal)[:, None, :, :]


class _VerifyShimCache:
    """Data-free cache passed to the model so shapes/offsets are well formed."""

    __slots__ = ("offset",)

    def __init__(self, offset: int):
        self.offset = int(offset)

    @property
    def state(self):
        return ()

    def make_mask(self, N, return_array=False, window_size=None, **kwargs):
        return None

    def update_and_fetch(self, keys, values):
        raise RuntimeError("_VerifyShimCache is only a shape placeholder")


def around_attention_call(original, wrapper, x, mask=None, cache=None):
    """Hook body for ``MLXAttentionWrapper.__call__``."""
    ctx = get_verify_context()
    if ctx is None:
        return original(wrapper, x, mask=mask, cache=cache)
    return _batched_verify_attention(wrapper, x, ctx)


def _batched_verify_attention(wrapper: Any, x: mx.array, ctx: VerifyContext):
    """Attention for ``B`` requests × ``T`` new tokens each.

    Mirrors ``MLXAttentionWrapper._batched_decode`` (projections, q/k norm,
    RoPE, SDPA, optional output gate) but treats the query axis as length T.
    """
    inner = wrapper._inner
    layer_idx = wrapper._layer_idx
    B, T, _ = x.shape
    if T != int(ctx.num_tokens):
        raise ValueError(
            f"verify token count {T} != context num_tokens {ctx.num_tokens}"
        )

    n_heads = int(getattr(wrapper, "_n_heads", inner.n_heads))
    n_kv_heads = int(getattr(wrapper, "_n_kv_heads", inner.n_kv_heads))
    scale = float(getattr(wrapper, "_scale", inner.scale))
    head_dim = getattr(wrapper, "_head_dim", None)

    q_proj_output = inner.q_proj(x)
    keys = inner.k_proj(x)
    values = inner.v_proj(x)
    if head_dim is None:
        head_dim = int(keys.shape[-1] // n_kv_heads)
    q_width = n_heads * head_dim

    gate = None
    if q_proj_output.shape[-1] == q_width:
        queries = q_proj_output.reshape(B, T, n_heads, head_dim)
    elif q_proj_output.shape[-1] == 2 * q_width:
        # Qwen3-Next / gated-attention variants.
        queries, gate = mx.split(
            q_proj_output.reshape(B, T, n_heads, 2 * head_dim), 2, axis=-1
        )
        gate = gate.reshape(B, T, q_width)
    else:
        raise RuntimeError(
            f"Unexpected q_proj output shape {q_proj_output.shape} for "
            f"{type(inner).__name__}"
        )

    keys = keys.reshape(B, T, n_kv_heads, head_dim)
    values = values.reshape(B, T, n_kv_heads, head_dim)

    if hasattr(inner, "q_norm"):
        queries = inner.q_norm(queries)
    if hasattr(inner, "k_norm"):
        keys = inner.k_norm(keys)

    queries = queries.transpose(0, 2, 1, 3)
    keys = keys.transpose(0, 2, 1, 3)
    values = values.transpose(0, 2, 1, 3)

    # [B] offsets: mx.fast.rope advances positions inside T automatically.
    queries = inner.rope(queries, offset=ctx.offsets)
    keys = inner.rope(keys, offset=ctx.offsets)

    # Append T rows to each request's cache and build a padded B×S view.
    max_len = ctx.max_len
    if _LEGACY_VERIFY:
        all_k: list[mx.array] = []
        all_v: list[mx.array] = []
        for i, lens in enumerate(ctx.prefix_lens):
            cache_i = ctx.caches[i][layer_idx]
            k_i, v_i = cache_i.update_and_fetch(
                keys[i : i + 1], values[i : i + 1]
            )
            pad = max_len - (lens + T)
            if pad > 0:
                k_pad = mx.zeros(
                    (1, n_kv_heads, pad, head_dim), dtype=k_i.dtype
                )
                v_pad = mx.zeros(
                    (1, n_kv_heads, pad, head_dim), dtype=v_i.dtype
                )
                k_i = mx.concatenate([k_i, k_pad], axis=2)
                v_i = mx.concatenate([v_i, v_pad], axis=2)
            all_k.append(k_i)
            all_v.append(v_i)
        keys_b = mx.concatenate(all_k, axis=0)
        values_b = mx.concatenate(all_v, axis=0)
        idx = mx.arange(max_len)
        valid = idx[None, :] < (ctx.offsets[:, None] + T)
        causal = idx[None, None, :] <= (
            ctx.offsets[:, None, None] + mx.arange(T)[None, :, None]
        )
        attn_mask = (valid[:, None, :] & causal)[:, None, :, :]
    else:
        # One zero buffer per layer plus direct slice writes; the legacy path
        # allocated and concatenated a pad tensor per request per layer.
        keys_b = mx.zeros((B, n_kv_heads, max_len, head_dim), dtype=keys.dtype)
        values_b = mx.zeros(
            (B, n_kv_heads, max_len, head_dim), dtype=values.dtype
        )
        for i, lens in enumerate(ctx.prefix_lens):
            cache_i = ctx.caches[i][layer_idx]
            cache_i.update_and_fetch(keys[i : i + 1], values[i : i + 1])
            end = int(lens) + T
            keys_b[i : i + 1, :, :end, :] = cache_i.keys[:, :, :end, :]
            values_b[i : i + 1, :, :end, :] = cache_i.values[:, :, :end, :]
        attn_mask = ctx.attn_mask

    sink_kwargs = getattr(wrapper, "_sink_kwargs", {}) or {}
    output = mx.fast.scaled_dot_product_attention(
        queries,
        keys_b,
        values_b,
        scale=scale,
        mask=attn_mask,
        **sink_kwargs,
    )
    output = output.transpose(0, 2, 1, 3).reshape(B, T, -1)
    if gate is not None:
        output = output * mx.sigmoid(gate)
    return inner.o_proj(output)


def _extract_logits(model_output):
    if isinstance(model_output, tuple):
        return model_output[0]
    return model_output


def run_batched_target_verify(
    model,
    caches: list[list[Any]],
    prefix_lens: Iterable[int],
    verify_tokens: mx.array,
    *,
    capture=None,
    num_layers: int,
    target_layer_ids=None,
) -> tuple[mx.array, mx.array]:
    """One target forward over ``[B,T]`` tokens; returns (logits, hidden).

    ``capture`` is a :class:`dspine_mlx.sglang.capture.CaptureSession`; when
    given, the configured target-layer outputs are collected as
    ``[B,T,sum(hidden)]`` in ``target_layer_ids`` order (falling back to call
    order).
    """
    prefix_lens = [int(x) for x in prefix_lens]
    B, T = (int(v) for v in verify_tokens.shape)
    if len(prefix_lens) != B:
        raise ValueError("prefix_lens length != verify batch size")
    shim = [_VerifyShimCache(max(prefix_lens)) for _ in range(num_layers)]
    ctx = VerifyContext(prefix_lens=prefix_lens, caches=caches, num_tokens=T)
    set_verify_context(ctx)
    if capture is not None:
        capture.begin(1)
    try:
        model_output = model(verify_tokens, cache=shim)
        logits = _extract_logits(model_output)
        records = capture.end() if capture is not None else None
        hidden = records[0] if records is not None else None
    finally:
        clear_verify_context()
        if capture is not None:
            capture.abort_if_active()
    mx.eval(logits)
    if hidden is not None:
        order = (
            [int(x) for x in target_layer_ids]
            if target_layer_ids is not None
            else list(hidden.keys())
        )
        mx.eval(*[hidden[lid] for lid in order])
        hidden = mx.concatenate([hidden[lid] for lid in order], axis=-1)
    return logits, hidden


def rollback_caches(caches: list[list[Any]], keep_lens: Iterable[int]) -> None:
    """Logically drop rows past each request's committed length.

    ``ContiguousAttentionKVCache`` only reads ``[:offset]``; the stale rows are
    never observed and the next verify overwrites them.
    """
    for i, keep in enumerate(int(x) for x in keep_lens):
        for cache in caches[i]:
            if cache is None:
                continue
            drop = int(cache.offset) - keep
            if drop < 0:
                raise ValueError(
                    f"cache offset {cache.offset} < keep {keep} for request {i}"
                )
            cache.offset = int(cache.offset) - drop
            local = getattr(cache, "_local", None)
            if local is not None:
                cache._local = max(0, int(local) - drop)


def gather_slots(req_to_token, req_pool_idx: int, start: int, length: int):
    """Scheduler KV slots for ``[start, start+length)`` of one request.

    Reserved for the radix-cache phase; v1 runs with ``--disable-radix-cache``.
    """
    return req_to_token[req_pool_idx, start : start + length]
