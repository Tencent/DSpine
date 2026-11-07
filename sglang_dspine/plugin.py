"""One plugin entry point for DSpine on SGLang 0.5.22."""

from sglang_dspine.spec import (
    ALGORITHM,
    DSpineSpecAlgo,
    handle_dspine_server_args,
    is_dspine,
    map_dspine_draft_architecture,
    worker_factory,
)

HOOK_TARGETS = {
    "handle_speculative_decoding": "sglang.srt.arg_groups.speculative_hook.handle_speculative_decoding",
    "spec_need_hidden_states": "sglang.srt.speculative.spec_utils.spec_need_hidden_states",
    "decode_cuda_graph_runner_cls": "sglang.srt.model_executor.model_runner.ModelRunner._decode_cuda_graph_runner_cls",
    "config_draft_model": "sglang.srt.configs.model_config.ModelConfig._config_draft_model",
    "dflash_sampling_verify": "sglang.srt.speculative.dflash_worker_v2.compute_dflash_sampling_correct_drafts_and_bonus",
}
_registered = False


def after_handle_speculative_decoding(result, server_args, *args, **kwargs):
    from sglang.srt.arg_groups.overrides import resolved_view

    if is_dspine(resolved_view(server_args).speculative_algorithm or ""):
        handle_dspine_server_args(server_args)
    return result


def around_spec_need_hidden_states(original, *args, **kwargs):
    from sglang.srt.runtime_context import get_spec

    if is_dspine(get_spec().speculative_algorithm or ""):
        return False
    return original(*args, **kwargs)


def around_decode_cuda_graph_runner_cls(original, model_runner, *args, **kwargs):
    if model_runner.is_draft_worker and is_dspine(model_runner.spec_algorithm):
        from sglang_dspine.runtime.graph_runner import DSpineDraftCudaGraphRunner

        return DSpineDraftCudaGraphRunner
    return original(model_runner, *args, **kwargs)


def around_config_draft_model(original, model_config, *args, **kwargs):
    map_dspine_draft_architecture(model_config)
    return original(model_config, *args, **kwargs)


def around_dflash_sampling_verify(original, *args, **kwargs):
    from sglang.srt.runtime_context import get_spec

    if is_dspine(get_spec().speculative_algorithm or ""):
        from sglang_dspine.runtime.worker import accept_dspine_sampled_round

        return accept_dspine_sampled_round(*args, **kwargs)
    return original(*args, **kwargs)


def around_seeded_sampler(
    original, sampler, probs, sampling_info, positions, simple_sampling_case
):
    from sglang.srt.runtime_context import get_spec

    if (
        is_dspine(get_spec().speculative_algorithm or "")
        and sampling_info.sampling_seed is not None
    ):
        from sglang_dspine.sampling.random import sample_prefill

        return sample_prefill(probs, sampling_info, positions)
    return original(sampler, probs, sampling_info, positions, simple_sampling_case)


def register():
    global _registered
    if _registered:
        return
    from sglang.srt.models.registry import ModelRegistry
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    from sglang.srt.speculative.spec_registry import get_spec as registered_spec

    from sglang_dspine.layers import head
    from sglang_dspine.layers.gdn import register_ragged_gdn_hooks
    from sglang_dspine.layers.layout import register_bf16_layout_hook
    from sglang_dspine.runtime.replayssm import hooks as replay
    from sglang_dspine.sampling import target as target_probs
    from sglang_dspine.verification.caps import register_verify_cap_hooks

    if "DSpineDraftModel" not in ModelRegistry.models:
        ModelRegistry.register("sglang_dspine.models", strict=True)
    hooks = (
        (
            "sglang.srt.layers.sampler.Sampler._sample_from_probs",
            around_seeded_sampler,
            HookType.AROUND,
        ),
        (
            HOOK_TARGETS["handle_speculative_decoding"],
            after_handle_speculative_decoding,
            HookType.AFTER,
        ),
        (
            HOOK_TARGETS["spec_need_hidden_states"],
            around_spec_need_hidden_states,
            HookType.AROUND,
        ),
        (
            HOOK_TARGETS["decode_cuda_graph_runner_cls"],
            around_decode_cuda_graph_runner_cls,
            HookType.AROUND,
        ),
        (
            HOOK_TARGETS["config_draft_model"],
            around_config_draft_model,
            HookType.AROUND,
        ),
        (
            HOOK_TARGETS["dflash_sampling_verify"],
            around_dflash_sampling_verify,
            HookType.AROUND,
        ),
        (
            "sglang.srt.model_executor.runner_backend.full_cuda_graph_backend.FullCudaGraphBackend.capture_one",
            head.around_capture,
            HookType.AROUND,
        ),
        (
            "sglang.srt.model_executor.runner_backend.full_cuda_graph_backend.FullCudaGraphBackend.cleanup",
            head.around_cleanup,
            HookType.AROUND,
        ),
        (
            "sglang.srt.managers.scheduler.Scheduler.__init__",
            target_probs.after_scheduler_init,
            HookType.AFTER,
        ),
    )
    for target, hook, kind in hooks:
        HookRegistry.register(target, hook, kind)
    register_verify_cap_hooks()
    register_ragged_gdn_hooks()
    register_bf16_layout_hook()
    replay.register()
    if registered_spec(ALGORITHM) is None:
        SpeculativeAlgorithm.register(
            ALGORITHM, supports_overlap=True, spec_class=DSpineSpecAlgo
        )(worker_factory)
    _registered = True
