"""Padding for target rows excluded by each request's verification length."""

_JOINT_SAMPLING_PARAMS = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "min_new_tokens": 0,
}


def _false_flags(value):
    return value is None or (
        isinstance(value, (list, tuple)) and all((x is False for x in value))
    )


def _supports_neutral_padding(controller):
    if getattr(controller, "sampled_round", None) is not True:
        return False
    if (
        getattr(getattr(controller, "worker", None), "sampled_policy", None)
        != "sampler"
    ):
        return False
    saved = getattr(controller, "saved_input", None)
    if not isinstance(saved, tuple) or len(saved) < 3:
        return False
    batch = saved[2]
    if getattr(batch, "has_grammar", True) is not False:
        return False
    if getattr(batch, "return_logprob", True) is not False:
        return False
    info = getattr(batch, "sampling_info", None)
    if info is None:
        return False
    for name, want in [
        ("is_all_greedy", False),
        ("is_any_greedy", False),
        ("need_top_k_sampling", True),
        ("need_top_p_sampling", True),
        ("need_min_p_sampling", False),
    ]:
        if getattr(info, name, None) is not want:
            return False
    if getattr(info, "has_custom_logit_processor", False):
        return False
    if getattr(info, "has_watermark_candidates", False):
        return False
    for name in (
        "custom_logit_processor",
        "logit_bias",
        "grammar_mask",
        "acc_additive_penalties",
        "acc_scaling_penalties",
        "acc_linear_penalties",
        "linear_penalties",
        "sampling_mask_batch_indices",
        "sampling_support_logprobs_capture_indices",
    ):
        if getattr(info, name, None) is not None:
            return False
    if any((x is not None for x in getattr(info, "grammars", None) or [])):
        return False
    for name in ("return_sampling_masks", "return_sampling_support_logprobs"):
        if not _false_flags(getattr(info, name, None)):
            return False
    if any((x is not None for x in getattr(info, "custom_params", None) or [])):
        return False
    penalizer = getattr(info, "penalizer_orchestrator", None)
    if penalizer is not None and getattr(penalizer, "is_required", None) is not False:
        return False
    reqs = getattr(batch, "reqs", None)
    if not isinstance(reqs, (list, tuple)) or not reqs:
        return False
    for req in reqs:
        if getattr(req, "grammar", None) is not None:
            return False
        for name in ("return_logprob", "return_sampling_mask"):
            if getattr(req, name, False) is not False:
                return False
        if getattr(req, "custom_logit_processor", None) is not None:
            return False
        params = getattr(req, "sampling_params", None)
        if params is None:
            return False
        for name, want in _JOINT_SAMPLING_PARAMS.items():
            value = getattr(params, name, None)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value != want
            ):
                return False
        for name in (
            "logit_bias",
            "json_schema",
            "regex",
            "ebnf",
            "structural_tag",
            "custom_params",
        ):
            if getattr(params, name, None) is not None:
                return False
    return True


def logits_padding(controller, value, shape):
    """Allocate a full block; the caller overwrites every verified row.

    A normalized dummy row avoids unnecessary nucleus work on unverified
    positions. Acceptance is bounded by verify_lens, so these rows cannot emit
    tokens. Other sampling modes retain zero padding.
    """
    if _supports_neutral_padding(controller):
        padded = value.new_full(shape, float("-inf"))
        padded[..., 0] = 0.0
        return padded
    return value.new_zeros(shape)
