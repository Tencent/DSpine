"""SGLang algorithm interface, server-argument validation, and checkpoint mapping.

DSPINE follows the DFlash-family scheduling and verification conventions.
"""

from typing import Type

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.speculative.spec_registry import CustomSpecAlgo

ALGORITHM = "DSPINE"
FP4_KV_CACHE_DTYPES = ("nvfp4", "fp4_mx_block16")


class DSpineSpecAlgo(CustomSpecAlgo):
    """DFlash-family behavior for SGLang's custom speculative algorithm interface.

    SGLang 0.5.20's scheduler also calls ``create_future_map``, ``need_topk``, and
    ``carries_draft_hidden_states``, which ``CustomSpecAlgo`` does not define.
    """

    def is_dflash_family(self) -> bool:
        return True

    def supports_target_verify_for_draft(self) -> bool:
        return True

    def supports_mixed_chunk(self) -> bool:
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
    from sglang_dspine.worker import DSpineWorkerV2

    return DSpineWorkerV2


def is_dspine(algorithm) -> bool:
    """True for the DSPINE algorithm, given its server-args name or the algorithm object."""
    if isinstance(algorithm, str):
        return algorithm.upper() == ALGORITHM
    return isinstance(algorithm, DSpineSpecAlgo)


def handle_dspine_server_args(server_args) -> None:
    """Validate DSPINE options, then apply DFlash argument resolution.

    SGLang gives registered algorithms a recording stand-in in ``handle_server_args``;
    ``_handle_dflash`` needs the real record to place its declarations correctly.
    The plugin calls this function after ``handle_speculative_decoding`` returns,
    while ``DSpineSpecAlgo`` keeps the inherited no-op ``handle_server_args``.
    """
    from sglang.srt.arg_groups.speculative_hook import _handle_dflash

    cfg = resolving_view(server_args)
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
            f"DSPINE does not support the {cfg.kv_cache_dtype} KV cache: its verify commits a "
            "prefix of a dense draft block, which is validated with the default KV cache only."
        )
    if cfg.enable_linear_replayssm_spec:
        raise ValueError("DSPINE does not support --enable-linear-replayssm-spec.")
    _handle_dflash(server_args)


def map_dspine_draft_architecture(model_config) -> None:
    """Map bridge-bearing DFlash/DSpine drafts to DSpineDraftModel under DSPINE.

    Called before SGLang's draft-config handling. Other algorithms reject these
    checkpoints because they would omit the code bridge.
    """
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
                "--speculative-algorithm DSPINE needs a DSpine draft "
                "(dflash_config.selected_prefix_code_bridge is missing)."
            )
        architectures[0] = "DSpineDraftModel"
    elif carries_bridge:
        raise ValueError(
            "This draft checkpoint carries a DSpine code bridge; serve it with "
            "--speculative-algorithm DSPINE."
        )
