"""Tiny random-weight hybrid (qwen3_5) target + DSpine draft for fast tests."""

from __future__ import annotations

import os

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
from mlx_lm.models import qwen3_5

from dspine_mlx.config import DraftConfig
from dspine_mlx.model import DSpineDraftModel

VOCAB = 512
HIDDEN = 128
TARGET_LAYERS = 8
TARGET_IDS = [1, 3, 5, 6, 7]
# 128 exercises the packed Metal GatedDeltaNet kernel (Qwen3.8-27B uses 128).
DK = int(os.environ.get("DSPINE_TINY_DK", "32"))


def target_config() -> dict:
    text = dict(
        model_type="qwen3_5_text",
        hidden_size=HIDDEN,
        intermediate_size=256,
        num_hidden_layers=TARGET_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        rms_norm_eps=1e-6,
        vocab_size=VOCAB,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=DK,
        linear_value_head_dim=DK,
        linear_conv_kernel_dim=4,
        full_attention_interval=4,
        tie_word_embeddings=False,
        max_position_embeddings=4096,
        rope_parameters=dict(
            rope_type="default", rope_theta=10000.0, partial_rotary_factor=0.25
        ),
    )
    return dict(model_type="qwen3_5", text_config=text)


def make_target(seed: int = 0):
    mx.random.seed(seed)
    args = qwen3_5.ModelArgs.from_dict(target_config())
    model = qwen3_5.Model(args)
    # Larger-than-default weights so logits have clear winners.
    model.update(
        tree_map(lambda p: p * 2.0 if p.ndim == 2 else p, model.parameters())
    )
    # Same rule as mlx-lm load: everything bf16 except A_log.
    from mlx.utils import tree_flatten

    model.load_weights(
        [
            (k, v if k.endswith("A_log") else v.astype(mx.bfloat16))
            for k, v in tree_flatten(model.parameters())
        ]
    )
    model.eval()
    mx.eval(model.parameters())
    return model


def draft_config(n_layers: int = 2, window: int | None = None) -> dict:
    raw = dict(
        architectures=["DFlashDraftModel"],
        model_type="qwen3",
        attention_bias=False,
        hidden_act="silu",
        hidden_size=HIDDEN,
        intermediate_size=256,
        num_hidden_layers=n_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        rms_norm_eps=1e-6,
        vocab_size=VOCAB,
        draft_vocab_size=VOCAB,
        max_position_embeddings=4096,
        rope_theta=10000000,
        rope_parameters=dict(rope_theta=10000000, rope_type="default"),
        rope_scaling=None,
        tie_word_embeddings=False,
        is_causal=True,
        block_size=16,
        num_target_layers=TARGET_LAYERS,
        code_bridge_second_pass=dict(
            first_exact=n_layers - 1, head_topk=16, mode="serial_table"
        ),
        dflash_config=dict(
            context_fusion_checkpointing=True,
            decoder_checkpointing=True,
            draft_block_causal=True,
            mask_token_id=VOCAB - 2,
            selected_prefix_code_bridge=dict(code_size=64, eps=1e-6, kappa=10.0, rank=32),
            selected_prefix_decode_mode="serial",
            selected_prefix_feature_layers=list(range(1, n_layers + 1)),
            selected_prefix_feature_write_mode="independent",
            selected_prefix_feature_writes=True,
            selected_prefix_last_write_refine=dict(
                head_topk=16, kind="exact_sender", version=1, weight=1.0
            ),
            selected_prefix_terminal=False,
            target_layer_ids=TARGET_IDS,
        ),
    )
    if window is not None:
        raw.update(
            layer_types=["sliding_attention"] * n_layers,
            sliding_window=window,
            use_sliding_window=True,
            max_window_layers=n_layers,
        )
    return raw


def make_draft(seed: int = 1, n_layers: int = 2, window: int | None = None):
    mx.random.seed(seed)
    cfg = DraftConfig.from_dict(draft_config(n_layers, window))
    model = DSpineDraftModel(cfg)
    br = model.selected_prefix_bridge
    br.read_weight = mx.random.normal(br.read_weight.shape) * 0.1
    br.early_read_weights = mx.random.normal(br.early_read_weights.shape) * 0.1
    br.feature_write_weights = mx.random.normal(br.feature_write_weights.shape) * 0.05
    br.codebook = mx.random.normal(br.codebook.shape)
    br.out.weight = mx.zeros_like(br.out.weight)
    br.independent_write_version = mx.array(1, dtype=mx.int64)
    br.terminal_disabled_version = mx.array(1, dtype=mx.int64)
    model.set_dtype(mx.bfloat16)
    br.prepare()
    mx.eval(model.parameters())
    return model, cfg


class FakeTokenizer:
    eos_token_ids = []
