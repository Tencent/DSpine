from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3 import MLP
from mlx_lm.models.rope_utils import initialize_rope

from .bridge import DSpineCodeBridge
from .config import DraftConfig, load_draft_config, resolve_draft_path


class DraftAttention(nn.Module):
    """Qwen3-style attention with a target-context K/V cache.

    Context K/V are computed from the shared fc-projected target features;
    block K/V are transient. DSpine requires causal attention inside the block
    while every context row stays visible.
    """

    def __init__(self, cfg: DraftConfig):
        super().__init__()
        self.n_heads = int(cfg.num_attention_heads)
        self.n_kv_heads = int(cfg.num_key_value_heads)
        self.head_dim = int(cfg.head_dim)
        self.scale = self.head_dim**-0.5
        dim = int(cfg.hidden_size)

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

    def __call__(self, x, x_ctx, rope, cache: KVCache | None = None):
        B, L, _ = x.shape
        S = 0 if x_ctx is None else x_ctx.shape[1]

        queries = self.q_proj(x)
        prop_keys = self.k_proj(x)
        prop_values = self.v_proj(x)

        queries = self.q_norm(
            queries.reshape(B, L, self.n_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        prop_keys = self.k_norm(
            prop_keys.reshape(B, L, self.n_kv_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        prop_values = prop_values.reshape(
            B, L, self.n_kv_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        if cache is not None:
            offset = cache.offset
            queries = rope(queries, offset=offset + S)
            prop_keys = rope(prop_keys, offset=offset + S)
            if S > 0:
                ctx_keys, ctx_values = self._context_kv(x_ctx, rope, offset)
                keys, values = cache.update_and_fetch(ctx_keys, ctx_values)
            elif cache.keys is None:
                keys = prop_keys
                values = prop_values
            else:
                keys = cache.keys[:, :, :offset, :]
                values = cache.values[:, :, :offset, :]
            keys = mx.concatenate([keys, prop_keys], axis=2)
            values = mx.concatenate([values, prop_values], axis=2)
        else:
            if x_ctx is None:
                raise ValueError(
                    "DraftAttention needs target context or a prefilled cache"
                )
            ctx_keys, ctx_values = self._context_kv(x_ctx, rope, 0)
            queries = rope(queries)
            prop_keys = rope(prop_keys)
            keys = mx.concatenate([ctx_keys, prop_keys], axis=2)
            values = mx.concatenate([ctx_values, prop_values], axis=2)

        output = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=self.scale,
            mask="causal",
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)

    def _context_kv(self, x_ctx, rope, offset: int):
        """Context K/V after k-norm and RoPE at absolute ``offset``."""
        B, S, _ = x_ctx.shape
        ctx_keys = self.k_proj(x_ctx)
        ctx_values = self.v_proj(x_ctx)
        ctx_keys = self.k_norm(
            ctx_keys.reshape(B, S, self.n_kv_heads, self.head_dim)
        ).transpose(0, 2, 1, 3)
        ctx_values = ctx_values.reshape(
            B, S, self.n_kv_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        return rope(ctx_keys, offset=offset), ctx_values

    def append_context(self, x_ctx, rope, cache: KVCache) -> None:
        """Append target-context K/V to ``cache`` without drafting a block.

        Used after prefill/verify so a later block-only forward sees exactly
        the cache state the combined forward would have produced.
        """
        if x_ctx is None or x_ctx.shape[1] == 0:
            return
        ctx_keys, ctx_values = self._context_kv(x_ctx, rope, cache.offset)
        cache.update_and_fetch(ctx_keys, ctx_values)


class DraftDecoderLayer(nn.Module):
    def __init__(self, cfg: DraftConfig):
        super().__init__()
        self.self_attn = DraftAttention(cfg)
        self.mlp = MLP(int(cfg.hidden_size), int(cfg.intermediate_size))
        self.input_layernorm = nn.RMSNorm(
            int(cfg.hidden_size), eps=cfg.rms_norm_eps
        )
        self.post_attention_layernorm = nn.RMSNorm(
            int(cfg.hidden_size), eps=cfg.rms_norm_eps
        )

    def __call__(self, x_normed, residual, x_ctx, rope, cache=None):
        # ``x_normed`` is the layer input already normalized (layer 0 by its
        # own input_layernorm, later layers by the fused bridge write).
        attn_out = self.self_attn(x_normed, x_ctx, rope, cache)
        h = residual + attn_out
        mlp_out = self.mlp(self.post_attention_layernorm(h))
        return h + mlp_out


class DSpineDraftModel(nn.Module):
    """DFlash-style block draft + DSpine code bridge.

    The forward returns ``(final_normed_block_hidden, stash)``. The refinement
    tail turns rows 1..T-1 into proposals and uses ``stash`` — the last write's
    pre-write raw / rms / packed state. ``target_hidden=None`` runs a block-only
    forward against an already-populated context cache.
    """

    def __init__(self, cfg: DraftConfig):
        super().__init__()
        self.config = cfg
        self.hidden_size = int(cfg.hidden_size)
        self.vocab_size = int(cfg.vocab_size)
        self.block_size = int(cfg.block_size)
        self.trained_block_size = int(cfg.trained_block_size)
        self.head_topk = int(cfg.head_topk)
        self.mask_token_id = int(cfg.mask_token_id)
        self.target_layer_ids = tuple(cfg.target_layer_ids)

        self.fc = nn.Linear(int(cfg.concat_size), self.hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(self.hidden_size, eps=cfg.rms_norm_eps)
        self.layers = [DraftDecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        self.norm = nn.RMSNorm(self.hidden_size, eps=cfg.rms_norm_eps)
        self.selected_prefix_bridge = DSpineCodeBridge(cfg)
        self.rope = initialize_rope(
            int(cfg.head_dim),
            base=float(cfg.rope_theta),
            traditional=False,
            scaling_config=None,
            max_position_embeddings=int(cfg.max_position_embeddings),
        )
        # Underscore-prefixed: borrowed from the target, never a parameter.
        self._embed_tokens: nn.Embedding | None = None

    def bind(self, target_model) -> "DSpineDraftModel":
        if hasattr(target_model, "embed_tokens"):
            inner = target_model
        elif hasattr(target_model, "model") and hasattr(
            target_model.model, "embed_tokens"
        ):
            inner = target_model.model
        elif (
            hasattr(target_model, "language_model")
            and hasattr(target_model.language_model, "model")
            and hasattr(target_model.language_model.model, "embed_tokens")
        ):
            inner = target_model.language_model.model
        else:
            raise AttributeError(
                f"Cannot find embed_tokens in {type(target_model).__name__}"
            )
        self._embed_tokens = inner.embed_tokens
        return self

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.layers]

    def set_block_size(self, block_size: int) -> None:
        self.config.validate_served_block_size(block_size)
        self.block_size = int(block_size)
        self.selected_prefix_bridge.set_block_size(block_size)

    def __call__(self, target_hidden, block_ids, caches=None):
        if self._embed_tokens is None:
            raise RuntimeError("Call bind(target_model) before the draft forward")
        B, T = block_ids.shape
        if caches is None:
            caches = [None] * len(self.layers)
        if len(caches) != len(self.layers):
            raise ValueError("draft cache length != number of draft layers")

        x = self._embed_tokens(block_ids)
        if target_hidden is not None:
            if target_hidden.shape[0] != B:
                raise ValueError(
                    f"target hidden batch {target_hidden.shape[0]} != block batch {B}"
                )
            ctx = self.hidden_norm(self.fc(target_hidden))
        else:
            # Context K/V is already in ``caches`` (see append_context).
            ctx = None

        residual = x
        x_in = self.layers[0].input_layernorm(x)
        last = len(self.layers) - 1
        stash_out: dict | None = None
        for i, layer in enumerate(self.layers):
            out = layer(x_in, residual, ctx, self.rope, caches[i])
            next_norm = self.layers[i + 1].input_layernorm if i < last else self.norm
            stash = {} if i == last else None
            x_in, residual = self.selected_prefix_bridge(
                out, None, block_ids, i, next_norm, stash
            )
            if stash is not None:
                stash_out = stash
        return x_in, stash_out

    def append_context(self, target_hidden, caches) -> None:
        """Append target-context K/V to every draft layer's cache.

        After this, a block-only forward (``target_hidden=None``) attends to
        exactly the context the combined forward would have built.
        """
        if target_hidden is None or target_hidden.shape[1] == 0:
            return
        if len(caches) != len(self.layers):
            raise ValueError("draft cache length != number of draft layers")
        ctx = self.hidden_norm(self.fc(target_hidden))
        for layer, cache in zip(self.layers, caches):
            if cache is None:
                raise ValueError("append_context requires per-layer caches")
            layer.self_attn.append_context(ctx, self.rope, cache)

    @property
    def num_parameters(self) -> int:
        return sum(p.size for _, p in tree_flatten(self.parameters()))


def _load_weights(path: Path) -> dict[str, mx.array]:
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no *.safetensors in {path}")
    weights: dict[str, mx.array] = {}
    for f in files:
        weights.update(mx.load(str(f)))
    return weights


def quantize_draft(
    model: DSpineDraftModel,
    *,
    bits: int = 8,
    group_size: int = 64,
    mode: str = "affine",
) -> DSpineDraftModel:
    """Quantize the draft backbone (fc + transformer layers); keep the bridge BF16.

    The bridge's codebook, read/write projections and norms stay in BF16 because
    the refinement tail depends on their exact rounding.
    """
    def predicate(path: str, module: nn.Module):
        if path.startswith("selected_prefix_bridge"):
            return False
        return isinstance(module, nn.Linear)

    nn.quantize(
        model,
        group_size=group_size,
        bits=bits,
        mode=mode,
        class_predicate=predicate,
    )
    return model


def load_draft(path: str | Path) -> tuple[DSpineDraftModel, DraftConfig]:
    """Load a DSpine draft checkpoint with strict name and shape checks."""
    root = resolve_draft_path(path)
    cfg = load_draft_config(root)
    model = DSpineDraftModel(cfg)
    weights = _load_weights(root)

    expected = dict(tree_flatten(model.parameters()))
    cleaned: dict[str, mx.array] = {}
    for name, tensor in weights.items():
        short = name.removeprefix("model.")
        if short in ("draft_block_causal_version", "last_write_refine_version"):
            if int(tensor.item()) != 1:
                raise ValueError(f"Unsupported {short}")
            continue
        cleaned[short] = tensor

    missing = sorted(set(expected) - set(cleaned))
    extra = sorted(set(cleaned) - set(expected))
    if missing or extra:
        raise ValueError(
            f"DSpine draft weight mismatch: missing={missing} extra={extra}"
        )
    for name, dest in expected.items():
        src = cleaned[name]
        if tuple(src.shape) != tuple(dest.shape):
            raise ValueError(
                f"DSpine draft shape mismatch for {name}: {src.shape} vs {dest.shape}"
            )
        if src.dtype != dest.dtype and dest.dtype in (mx.int32, mx.int64):
            cleaned[name] = src.astype(dest.dtype)
    model.load_weights(list(cleaned.items()), strict=True)
    model.selected_prefix_bridge.prepare()
    return model, cfg
