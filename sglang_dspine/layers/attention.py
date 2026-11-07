"""DSpine serving on SGLang 0.5.22."""

WINDOW = 8192
BLOCK_SIZE = 16
NUM_LAYERS = 5
BACKENDS = frozenset(("fa3",))


def _get(config, name, default=None):
    return (
        config.get(name, default)
        if isinstance(config, dict)
        else getattr(config, name, default)
    )


def validate_native_attention_config(config):
    """Return W for the exact configuration, None for the preexisting full path.

    Never infer sliding layers from a window field: SGLang DFlash reads
    layer_types, so a contradictory declaration must fail before model creation.
    """
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    types = _get(text, "layer_types", _get(config, "layer_types"))
    window = _get(text, "sliding_window", _get(config, "sliding_window"))
    use_swa = _get(
        text, "use_sliding_window", _get(config, "use_sliding_window", False)
    )
    declares_swa = types is not None and "sliding_attention" in types or bool(use_swa)
    if not declares_swa:
        return None
    if (
        _get(text, "num_hidden_layers") != NUM_LAYERS
        or types != ["sliding_attention"] * NUM_LAYERS
        or type(window) is not int
        or (window != WINDOW)
        or (type(_get(config, "block_size")) is not int)
        or (_get(config, "block_size") != BLOCK_SIZE)
    ):
        raise ValueError(
            "Native DSPINE SWA configuration requires all 5 sliding_attention layers, W8192 and trained B16"
        )
    bridge = _get(config, "dflash_config", {}) or {}
    if (
        _get(text, "is_causal") is not True
        or _get(bridge, "draft_block_causal") is not True
    ):
        raise ValueError(
            "Native DSPINE SWA configuration requires explicit is_causal=true and draft_block_causal=true"
        )
    if _get(text, "is_hybrid_swa", False):
        raise ValueError(
            "Native DSPINE SWA configuration retains dense KV; hybrid SWA pools are unvalidated"
        )
    return WINDOW


def validate_native_layer_windows(layers, window, *, decoder_type):
    """Check instantiated parser output, including the duplicate RadixAttention field."""
    expected_left = -1 if window is None else window - 1
    for index, layer in enumerate(layers):
        attention = layer.self_attn
        if attention.attn.attn_type != decoder_type:
            raise ValueError(
                f"DSPINE requires causal DECODER attention at layer {index}"
            )
        if (
            attention.sliding_window_size != expected_left
            or attention.attn.sliding_window_size != expected_left
        ):
            raise ValueError(
                f"DSPINE layer {index} must use window_left={expected_left} in both DFlash and RadixAttention"
            )


def validate_native_swa_runner(runner, *, served_block_size):
    """Fail before graph capture unless the audited native backend is active."""
    if served_block_size != BLOCK_SIZE:
        raise ValueError("Qwen3.8-27B DSpine serves its trained block size16")
    if runner.draft_attention_backend not in BACKENDS:
        raise ValueError(
            "Native DSPINE SWA configuration requires explicit --speculative-draft-attention-backend fa3"
        )
    if runner.sliding_window_size != WINDOW - 1:
        raise ValueError("Native DSPINE runner must preserve window_left=8191")
    if runner.is_hybrid_swa:
        raise ValueError(
            "Native DSPINE SWA configuration requires the dense draft KV path"
        )
