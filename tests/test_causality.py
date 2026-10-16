"""Checkpoint causality must reach DFlash before it constructs attention layers."""

from unittest import mock

import pytest
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.models.dflash import (
    DFlashDraftModel,
    _get_dflash_layer_attention_params,
)
from transformers import Qwen3Config

from sglang_dspine.models.dspine import DSpineDraftModel

_MISSING = object()


class _ParentConstructionReached(Exception):
    pass


@pytest.mark.parametrize(
    "draft_block_causal,is_causal,expected_attention_type",
    [
        pytest.param(True, _MISSING, AttentionType.DECODER, id="causal-missing"),
        pytest.param(True, None, AttentionType.DECODER, id="causal-none"),
        pytest.param(True, True, AttentionType.DECODER, id="explicit-causal"),
        pytest.param(True, False, AttentionType.ENCODER_ONLY, id="explicit-noncausal"),
        pytest.param(
            _MISSING, _MISSING, AttentionType.ENCODER_ONLY, id="missing-checkpoint-flag"
        ),
        pytest.param(
            False, _MISSING, AttentionType.ENCODER_ONLY, id="noncausal-checkpoint"
        ),
    ],
)
def test_checkpoint_causality_is_resolved_before_dflash_initialization(
    draft_block_causal, is_causal, expected_attention_type
):
    dflash_config = {}
    if draft_block_causal is not _MISSING:
        dflash_config["draft_block_causal"] = draft_block_causal
    config = Qwen3Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        layer_types=["full_attention"],
        dflash_config=dflash_config,
    )
    if is_causal is not _MISSING:
        config.is_causal = is_causal

    def inspect_parent_config(_model, parent_config, quant_config=None, prefix=""):
        assert parent_config is config
        assert _get_dflash_layer_attention_params(parent_config, 0) == (
            -1,
            expected_attention_type,
        )
        raise _ParentConstructionReached

    with (
        mock.patch.object(
            DFlashDraftModel,
            "__init__",
            autospec=True,
            side_effect=inspect_parent_config,
        ) as parent_init,
        pytest.raises(_ParentConstructionReached),
    ):
        DSpineDraftModel(config)
    parent_init.assert_called_once()
