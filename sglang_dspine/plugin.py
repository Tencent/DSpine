"""Register DSpine's speculative algorithm, draft model, and SGLang hooks.

SGLang loads ``register`` through the ``sglang.srt.plugins`` entry point.
The hooks adapt SGLang 0.5.20; registration is idempotent.
"""

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
}

_registered = False


def after_handle_speculative_decoding(result, server_args, *args, **kwargs):
    from sglang.srt.arg_groups.overrides import resolving_view

    if is_dspine(resolving_view(server_args).speculative_algorithm or ""):
        handle_dspine_server_args(server_args)
    return result


def around_spec_need_hidden_states(original, *args, **kwargs):
    from sglang.srt.runtime_context import get_spec

    if is_dspine(get_spec().speculative_algorithm or ""):
        return False
    return original(*args, **kwargs)


def around_decode_cuda_graph_runner_cls(original, model_runner, *args, **kwargs):
    if model_runner.is_draft_worker and is_dspine(model_runner.spec_algorithm):
        from sglang_dspine.graph_runner import DSpineDraftCudaGraphRunner

        return DSpineDraftCudaGraphRunner
    return original(model_runner, *args, **kwargs)


def around_config_draft_model(original, model_config, *args, **kwargs):
    map_dspine_draft_architecture(model_config)
    return original(model_config, *args, **kwargs)


def register() -> None:
    global _registered
    if _registered:
        return
    from sglang.srt.models.registry import ModelRegistry
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    from sglang.srt.speculative.spec_registry import get_spec as get_registered_spec

    if get_registered_spec(ALGORITHM) is None:
        SpeculativeAlgorithm.register(
            ALGORITHM, supports_overlap=True, spec_class=DSpineSpecAlgo
        )(worker_factory)
    if "DSpineDraftModel" not in ModelRegistry.models:
        ModelRegistry.register("sglang_dspine.models", strict=True)
    HookRegistry.register(
        HOOK_TARGETS["handle_speculative_decoding"],
        after_handle_speculative_decoding,
        HookType.AFTER,
    )
    HookRegistry.register(
        HOOK_TARGETS["spec_need_hidden_states"],
        around_spec_need_hidden_states,
        HookType.AROUND,
    )
    HookRegistry.register(
        HOOK_TARGETS["decode_cuda_graph_runner_cls"],
        around_decode_cuda_graph_runner_cls,
        HookType.AROUND,
    )
    HookRegistry.register(
        HOOK_TARGETS["config_draft_model"], around_config_draft_model, HookType.AROUND
    )
    _registered = True
