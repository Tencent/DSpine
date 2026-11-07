"""DSpine serving on SGLang 0.5.22."""

from typing import Type

from sglang.srt.arg_groups.overrides import resolved_view
from sglang.srt.speculative.spec_registry import CustomSpecAlgo

ALGORITHM = "DSPINE"
FP4_KV_CACHE_DTYPES = ("nvfp4", "fp4_mx_block16")


class DSpineSpecAlgo(CustomSpecAlgo):
    def supports_ragged_verify(self) -> bool:
        from sglang_dspine.verification.ragged import ragged_active

        return ragged_active()

    def is_dflash_family(self) -> bool:
        return True

    def supports_target_verify_for_draft(self) -> bool:
        return True

    def supports_grammar_overlap(self) -> bool:
        return True

    def need_topk(self) -> bool:
        return False

    def carries_draft_hidden_states(self) -> bool:
        return False

    def create_future_map(
        self,
        device,
        req_to_token_pool,
        needs_cpu_seq_lens: bool = True,
        needs_confidence_relay: bool = False,
    ):
        from sglang.srt.managers.overlap_utils import FutureMap

        return FutureMap(
            device, self, req_to_token_pool, needs_cpu_seq_lens, needs_confidence_relay
        )


def worker_factory(server_args) -> Type:
    from sglang_dspine.runtime.worker import DSpineWorkerV2

    return DSpineWorkerV2


def is_dspine(algorithm) -> bool:
    """True for the DSPINE algorithm, given its server-args name or the algorithm object."""
    if isinstance(algorithm, str):
        return algorithm.upper() == ALGORITHM
    return isinstance(algorithm, DSpineSpecAlgo)


def handle_dspine_server_args(server_args) -> None:
    from sglang.srt.arg_groups.speculative_hook import _handle_dflash

    cfg = resolved_view(server_args)
    if not str(cfg.device).startswith("cuda"):
        raise ValueError(
            f"DSPINE speculative decoding requires CUDA, got device={cfg.device!r}."
        )
    if int(cfg.tp_size) != 1:
        raise ValueError("DSPINE speculative decoding supports tp_size == 1 only.")
    if cfg.speculative_draft_window_size is not None:
        raise ValueError("DSPINE does not support --speculative-draft-window-size.")
    if cfg.disaggregation_mode != "null":
        raise ValueError("DSPINE does not support prefill/decode disaggregation.")
    if cfg.kv_cache_dtype in FP4_KV_CACHE_DTYPES:
        raise ValueError(
            f"DSPINE does not support the {cfg.kv_cache_dtype} KV cache: its verify commits a prefix of a dense draft block, which is validated with the default KV cache only."
        )
    if cfg.enable_linear_replayssm_spec:
        if int(cfg.pp_size) != 1:
            raise ValueError("DSPine ReplaySSM port requires pp_size == 1.")
        if cfg.enable_unified_memory or cfg.enable_page_major_kv_layout:
            raise ValueError(
                "DSPINE ReplaySSM requires the standard non-unified Mamba pool."
            )
        if cfg.disable_radix_cache:
            raise ValueError("DSPINE prefix cache requires radix caching enabled.")
        if cfg.mamba_radix_cache_strategy != "extra_buffer":
            raise ValueError(
                "DSPINE ReplaySSM prefix caching requires extra_buffer checkpoints."
            )
        if cfg.mamba_ssm_dtype not in (None, "float32"):
            raise ValueError("DSPINE ReplaySSM requires --mamba-ssm-dtype float32.")
        if cfg.enable_linear_replayssm:
            raise ValueError(
                "DSPINE spec ReplaySSM cannot also enable decode ReplaySSM."
            )
        if cfg.linear_attn_backend != "triton":
            raise ValueError("DSPINE ReplaySSM requires --linear-attn-backend triton.")
        for name in (
            "linear_attn_prefill_backend",
            "linear_attn_decode_backend",
            "linear_attn_verify_backend",
        ):
            if getattr(cfg, name) not in (None, "triton"):
                raise ValueError(
                    "DSPINE ReplaySSM requires Triton linear-attention backends."
                )
    _handle_dflash(server_args)


def map_dspine_draft_architecture(model_config) -> None:
    hf_config = model_config.hf_config
    architectures = getattr(hf_config, "architectures", None)
    if not model_config.is_draft_model or not architectures:
        return
    if architectures[0] not in ("DFlashDraftModel", "DSpineDraftModel"):
        return
    dflash_config = getattr(hf_config, "dflash_config", None) or {}
    carries_bridge = bool(dict(dflash_config).get("selected_prefix_code_bridge"))
    if is_dspine(model_config.speculative_algorithm or ""):
        if not carries_bridge:
            raise ValueError(
                "--speculative-algorithm DSPINE needs a DSpine draft (dflash_config.selected_prefix_code_bridge is missing)."
            )
        architectures[0] = "DSpineDraftModel"
        is_causal = getattr(hf_config, "is_causal", None)
        if dict(dflash_config).get("draft_block_causal"):
            if is_causal is False:
                raise ValueError(
                    "DSpine draft config contradicts itself: dflash_config.draft_block_causal is true but is_causal is false."
                )
            if is_causal is None:
                hf_config.is_causal = True
    elif carries_bridge:
        raise ValueError(
            "This draft checkpoint carries a DSpine code bridge; serve it with --speculative-algorithm DSPINE."
        )
