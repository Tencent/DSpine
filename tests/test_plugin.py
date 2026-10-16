"""CPU tests for DSPINE registration and hooks against SGLang 0.5.20.
Verify algorithm behavior, draft architecture registration, and applied hooks."""

import types
from unittest import mock

import pytest
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang.srt.speculative import spec_utils
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

from sglang_dspine import plugin
from sglang_dspine.graph_runner import DSpineDraftCudaGraphRunner
from sglang_dspine.models.dspine import DSpineDraftModel
from sglang_dspine.spec import (
    DSpineSpecAlgo,
    handle_dspine_server_args,
    map_dspine_draft_architecture,
)
from sglang_dspine.worker import DSpineWorkerV2


@pytest.fixture(scope="module")
def dspine():
    plugin.register()
    plugin.register()  # idempotent
    HookRegistry.apply_hooks()
    return SpeculativeAlgorithm.from_string("dspine")


def test_algorithm_answers_like_a_dflash_family_member(dspine):
    assert isinstance(dspine, DSpineSpecAlgo) and dspine.name == "DSPINE"
    # SGLang's enum, for a DFlash-family member that is neither DFLASH nor DSPARK:
    # is_dflash_family True; target verify for the draft and grammar overlap follow the family;
    # mixed chunk allowed (as DFLASH); ragged verify is DSPARK-only; draft KV unless NGRAM; top-k
    # and carried draft hidden states are EAGLE-family only.
    expected = dict(
        is_some=True,
        is_none=False,
        is_speculative=True,
        is_eagle=False,
        is_eagle3=False,
        is_frozen_kv_mtp=False,
        is_dflash=False,
        is_uno=False,
        is_dspark=False,
        is_dflash_family=True,
        is_standalone=False,
        is_ngram=False,
        supports_target_verify_for_draft=True,
        supports_mixed_chunk=True,
        supports_ragged_verify=False,
        supports_grammar_overlap=True,
        has_draft_kv=True,
        need_topk=False,
        carries_draft_hidden_states=False,
    )
    for name, value in expected.items():
        assert getattr(dspine, name)() is value, name
    # target verify covers the whole block on both runners (DSPARK alone drops one on the draft)
    assert dspine.get_num_tokens_per_req_for_target_verify(16, True) == 16
    assert dspine.get_num_tokens_per_req_for_target_verify(16, False) == 16
    assert dspine.supports_overlap
    assert dspine.factory(None) is DSpineWorkerV2


def test_future_map_is_built_like_the_enum_builds_it(dspine):
    with mock.patch("sglang.srt.managers.overlap_utils.FutureMap") as future_map:
        dspine.create_future_map(
            "cuda:0", "pool", needs_cpu_seq_lens=False, needs_confidence_relay=True
        )
    future_map.assert_called_once_with("cuda:0", dspine, "pool", False, True)


def test_draft_model_architecture_is_registered(dspine):
    assert ModelRegistry.models["DSpineDraftModel"] is DSpineDraftModel


def test_hooks_are_registered_on_their_targets(dspine):
    for key, hook_type, hook in (
        (
            "handle_speculative_decoding",
            HookType.AFTER,
            plugin.after_handle_speculative_decoding,
        ),
        (
            "spec_need_hidden_states",
            HookType.AROUND,
            plugin.around_spec_need_hidden_states,
        ),
        (
            "decode_cuda_graph_runner_cls",
            HookType.AROUND,
            plugin.around_decode_cuda_graph_runner_cls,
        ),
        ("config_draft_model", HookType.AROUND, plugin.around_config_draft_model),
    ):
        entries = HookRegistry._hooks[plugin.HOOK_TARGETS[key]]
        assert [(t, h) for t, h, _ in entries] == [(hook_type, hook)], key


def test_draft_graph_runner_is_selected_for_the_dspine_draft_runner_only(dspine):
    def runner_cls(algorithm, is_draft_worker):
        model_runner = types.SimpleNamespace(
            spec_algorithm=algorithm, is_draft_worker=is_draft_worker
        )
        return ModelRunner._decode_cuda_graph_runner_cls(model_runner)

    assert runner_cls(dspine, True) is DSpineDraftCudaGraphRunner
    assert runner_cls(dspine, False) is DecodeCudaGraphRunner
    assert runner_cls(SpeculativeAlgorithm.DFLASH, True) is DecodeCudaGraphRunner


def test_no_hidden_states_relay_for_dspine(dspine):
    # SGLang: STANDALONE / DFLASH / DSPARK relay none; otherwise not with multi-layer EAGLE
    for algorithm, expected in (("DSPINE", False), ("DFLASH", False), ("EAGLE", True)):
        spec = types.SimpleNamespace(
            speculative_algorithm=algorithm, enable_multi_layer_eagle=False
        )
        with (
            mock.patch("sglang.srt.runtime_context.get_spec", return_value=spec),
            mock.patch.object(spec_utils, "get_spec", return_value=spec),
        ):
            assert spec_utils.spec_need_hidden_states() is expected, algorithm


def test_server_args_hook_resolves_dspine_only(dspine):
    resolved = []
    with mock.patch.object(plugin, "handle_dspine_server_args", resolved.append):
        for algorithm in ("DSPINE", "dspine", "DFLASH", None):
            server_args = types.SimpleNamespace(speculative_algorithm=algorithm)
            result = object()
            assert (
                plugin.after_handle_speculative_decoding(result, server_args) is result
            )
    assert [args.speculative_algorithm for args in resolved] == ["DSPINE", "dspine"]


def _server_args(**overrides):
    fields = dict(
        device="cuda",
        tp_size=1,
        speculative_draft_window_size=None,
        disaggregation_mode="null",
        kv_cache_dtype="auto",
        enable_linear_replayssm_spec=False,
    )
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


def test_worker_constructor_matches_the_installed_base():
    # SGLang 0.5.20 passes ParallelState (`ps`) when constructing draft workers.
    # Keep the constructor signature aligned with DFlashWorkerV2.
    import inspect

    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    from sglang_dspine.worker import DSpineWorkerV2

    base = list(inspect.signature(DFlashWorkerV2.__init__).parameters)
    assert list(inspect.signature(DSpineWorkerV2.__init__).parameters) == base
    assert "ps" in base


def test_server_args_fields_exist_in_this_sglang():
    from sglang.srt.arg_groups.overrides import field_names
    from sglang.srt.server_args import ServerArgs

    assert set(vars(_server_args())) <= set(field_names(ServerArgs))


def test_dspine_envelope_then_dflash_resolution():
    with mock.patch("sglang.srt.arg_groups.speculative_hook._handle_dflash") as dflash:
        ok = _server_args()
        handle_dspine_server_args(ok)
        dflash.assert_called_once_with(ok)
        for bad in (
            dict(device="cpu"),
            dict(tp_size=2),
            dict(speculative_draft_window_size=512),
            dict(disaggregation_mode="decode"),
            dict(kv_cache_dtype="nvfp4"),
            dict(kv_cache_dtype="fp4_mx_block16"),
            dict(enable_linear_replayssm_spec=True),
        ):
            with pytest.raises(ValueError):
                handle_dspine_server_args(_server_args(**bad))
        assert dflash.call_count == 1


def _model_config(architecture, algorithm, bridge, is_draft=True):
    dflash_config = {"selected_prefix_code_bridge": {"code_size": 8}} if bridge else {}
    hf_config = types.SimpleNamespace(
        architectures=[architecture], dflash_config=dflash_config
    )
    return types.SimpleNamespace(
        hf_config=hf_config, is_draft_model=is_draft, speculative_algorithm=algorithm
    )


def test_draft_architecture_mapping():
    for architecture in ("DFlashDraftModel", "DSpineDraftModel"):
        config = _model_config(architecture, "DSPINE", bridge=True)
        map_dspine_draft_architecture(config)
        assert config.hf_config.architectures == ["DSpineDraftModel"]
    refused = (
        ("DFlashDraftModel", "DSPINE", False),  # DSPINE needs a code bridge
        ("DSpineDraftModel", "DSPINE", False),
        ("DFlashDraftModel", "DFLASH", True),  # DFLASH would drop the bridge
        ("DSpineDraftModel", "EAGLE", True),
    )
    for architecture, algorithm, bridge in refused:
        with pytest.raises(ValueError):
            map_dspine_draft_architecture(
                _model_config(architecture, algorithm, bridge)
            )
    untouched = (
        (_model_config("DFlashDraftModel", "DFLASH", False), "DFlashDraftModel"),
        (_model_config("DFlashDraftModel", "DSPINE", True, False), "DFlashDraftModel"),
        (_model_config("LlamaForCausalLM", "DSPINE", True), "LlamaForCausalLM"),
        (_model_config("DFlashDraftModel", None, False), "DFlashDraftModel"),
    )
    for config, architecture in untouched:
        map_dspine_draft_architecture(config)
        assert config.hf_config.architectures == [architecture]


def test_config_draft_model_hook_maps_before_sglang_runs(dspine):
    seen = []
    config = _model_config("DFlashDraftModel", "DSPINE", bridge=True)
    plugin.around_config_draft_model(
        lambda c: seen.append(list(c.hf_config.architectures)), config
    )
    assert seen == [["DSpineDraftModel"]]
