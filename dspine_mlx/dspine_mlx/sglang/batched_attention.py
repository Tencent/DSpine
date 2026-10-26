"""Batched draft-backbone forward for the SGLang MLX worker.

``dspine_mlx.model`` stays untouched.  This module reproduces its five-layer
block-only forward with one attention call over ``[B, T, H]`` and per-request
padded context KV (see :class:`dspine_mlx.sglang.batch.DraftBatchContext`).
"""

from __future__ import annotations

import mlx.core as mx


def attention_block_batched(attn, x, rope, ctx, layer_index: int):
    """Block-only attention for ``B`` requests with padded context."""
    B, L, _ = x.shape
    if L != int(ctx.num_tokens):
        raise ValueError(
            f"batched block has {L} tokens, expected {ctx.num_tokens}"
        )
    head_dim = attn.head_dim
    queries = attn.q_proj(x)
    prop_keys = attn.k_proj(x)
    prop_values = attn.v_proj(x)

    queries = attn.q_norm(
        queries.reshape(B, L, attn.n_heads, head_dim)
    ).transpose(0, 2, 1, 3)
    prop_keys = attn.k_norm(
        prop_keys.reshape(B, L, attn.n_kv_heads, head_dim)
    ).transpose(0, 2, 1, 3)
    prop_values = prop_values.reshape(
        B, L, attn.n_kv_heads, head_dim
    ).transpose(0, 2, 1, 3)

    queries = rope(queries, offset=ctx.offsets)
    prop_keys = rope(prop_keys, offset=ctx.offsets)

    caches = ctx.layer_caches(layer_index)
    max_prefix = ctx.max_prefix
    key_parts = []
    value_parts = []
    for b in range(B):
        cache = caches[b]
        n = int(ctx.prefix_lens[b])
        if n > 0 and getattr(cache, "keys", None) is not None:
            keys_b = cache.keys[:, :, :n, :]
            values_b = cache.values[:, :, :n, :]
        else:
            keys_b = mx.zeros(
                (1, attn.n_kv_heads, 0, head_dim), dtype=prop_keys.dtype
            )
            values_b = mx.zeros(
                (1, attn.n_kv_heads, 0, head_dim), dtype=prop_values.dtype
            )
        pad = max_prefix - n
        if pad > 0:
            keys_b = mx.concatenate(
                [
                    keys_b,
                    mx.zeros(
                        (1, attn.n_kv_heads, pad, head_dim),
                        dtype=keys_b.dtype,
                    ),
                ],
                axis=2,
            )
            values_b = mx.concatenate(
                [
                    values_b,
                    mx.zeros(
                        (1, attn.n_kv_heads, pad, head_dim),
                        dtype=values_b.dtype,
                    ),
                ],
                axis=2,
            )
        key_parts.append(
            mx.concatenate([keys_b, prop_keys[b : b + 1]], axis=2)
        )
        value_parts.append(
            mx.concatenate([values_b, prop_values[b : b + 1]], axis=2)
        )

    keys = mx.concatenate(key_parts, axis=0)
    values = mx.concatenate(value_parts, axis=0)
    output = mx.fast.scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale=attn.scale,
        mask=ctx.mask,
    )
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return attn.o_proj(output)


def forward_block_batched(model, block_ids: mx.array, ctx):
    """Run the draft backbone on ``[B, T]`` block ids against prefilled caches.

    Returns ``(final_normed_block_hidden [B,T,H], stash)`` exactly like
    ``DSpineDraftModel(target_hidden=None, block_ids, caches=...)``.
    """
    if model._embed_tokens is None:
        raise RuntimeError("bind the draft to the target model first")
    B, T = block_ids.shape
    if T != int(ctx.num_tokens):
        raise ValueError(
            f"block has {T} tokens, context expects {ctx.num_tokens}"
        )
    x = model._embed_tokens(block_ids)
    residual = x
    x_in = model.layers[0].input_layernorm(x)
    last = len(model.layers) - 1
    stash_out = None
    for i, layer in enumerate(model.layers):
        attn_out = attention_block_batched(
            layer.self_attn, x_in, model.rope, ctx, i
        )
        h = residual + attn_out
        mlp_out = layer.mlp(layer.post_attention_layernorm(h))
        out = h + mlp_out
        next_norm = (
            model.layers[i + 1].input_layernorm if i < last else model.norm
        )
        stash = {} if i == last else None
        x_in, residual = model.selected_prefix_bridge(
            out, None, block_ids, i, next_norm, stash
        )
        if stash is not None:
            stash_out = stash
    return x_in, stash_out
