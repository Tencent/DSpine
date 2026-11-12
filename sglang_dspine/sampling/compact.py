"""Request-wise logit adjustments for packed verification rows."""

import torch


def can_keep_compact(info):
    return (
        info is not None
        and not getattr(info, "has_custom_logit_processor", False)
        and getattr(info, "sampling_mask_batch_indices", None) is None
    )


def adjust_logits(logits, info, row_to_request):
    """Apply native transforms to real logits, in request order.

    Packed logits use ``row_to_request``; full blocks are viewed as [B, T, V]
    with no mapping. Multiplicative penalties must see the actual logit sign,
    rather than a zero-filled bias tensor.
    """
    if info is None:
        return

    def mapped(value):
        value = value.to(device=logits.device)
        return (
            value.index_select(0, row_to_request)
            if row_to_request is not None
            else value[:, None, :]
        )

    def scale(value):
        penalties = mapped(value)
        logits.copy_(torch.where(logits < 0, logits * penalties, logits / penalties))

    # The pinned SamplingBatchInfo calls these additive penalties. Older
    # DFlash callers use the linear name for the same precomputed buffer.
    linear = getattr(info, "acc_additive_penalties", None)
    if linear is None:
        linear = getattr(info, "acc_linear_penalties", None)
    if linear is not None:
        logits.add_(mapped(linear))
    scaling = getattr(info, "acc_scaling_penalties", None)
    if scaling is not None:
        scale(scaling)

    penalizer = getattr(info, "penalizer_orchestrator", None)
    if penalizer is not None and penalizer.is_required:
        # Match orchestrator.apply's ordering, including when an additive
        # penalizer follows a multiplicative one in non-overlap scheduling.
        for penalty in penalizer.penalizers.values():
            if not penalty.is_prepared():
                continue
            if penalty.is_multiplicative:
                scale(penalty.get_scaling_penalties())
            else:
                bias = torch.zeros(
                    (len(info), logits.shape[-1]),
                    device=logits.device,
                    dtype=torch.float32,
                )
                penalty.apply(bias)
                logits.add_(mapped(bias))

    grammar = getattr(info, "grammar_mask", None)
    if grammar is not None:
        mask = grammar.vocab_mask.to(device=logits.device)
        mask = (
            mask.index_select(0, row_to_request)
            if row_to_request is not None
            else mask.repeat_interleave(logits.shape[1], dim=0)
        )
        grammar.grammar.apply_vocab_mask(
            logits=logits.reshape(-1, logits.shape[-1]), vocab_mask=mask
        )

    bias = getattr(info, "logit_bias", None)
    if bias is not None:
        logits.add_(mapped(bias))


def around_verify_logits_adjustments(original, *, next_token_logits, sampling_info,
                                     draft_token_num):
    from sglang_dspine.verification.caps import active_controller

    controller = active_controller()
    if controller is None:
        from sglang_dspine.runtime import worker

        # Generic DSpine models have no optimized verification controller.
        if worker._active_worker is None:
            return original(
                next_token_logits=next_token_logits,
                sampling_info=sampling_info,
                draft_token_num=draft_token_num,
            )
    if controller is not None and getattr(controller, "compact_round", False):
        if next_token_logits is not controller.compact_logits:
            raise RuntimeError("Packed verification logits do not belong to this round")
        return adjust_logits(next_token_logits, sampling_info, controller.row_to_request)
    if sampling_info is None:
        return
    if next_token_logits.ndim != 2 or draft_token_num <= 0:
        raise ValueError("Verification needs 2D logits and a positive draft width")
    bs = len(sampling_info)
    if next_token_logits.shape[0] != bs * draft_token_num:
        raise ValueError("Verification logits do not match the request/block layout")
    if getattr(sampling_info, "has_custom_logit_processor", False):
        from sglang.srt.layers.sampler import apply_custom_logit_processor

        apply_custom_logit_processor(
            next_token_logits, sampling_info, num_tokens_in_batch=draft_token_num
        )
    return adjust_logits(
        next_token_logits.view(bs, draft_token_num, -1), sampling_info, None
    )
