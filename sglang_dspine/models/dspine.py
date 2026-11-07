"""DSpine serving on SGLang 0.5.22."""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Tuple

import torch
import triton
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.dflash import DFlashDecoderLayer, DFlashDraftModel
from sglang.srt.speculative.dflash_utils import _get_dflash_config
from torch import nn
from torch.nn import functional as F

from sglang_dspine.kernels import dspine as kernels
from sglang_dspine.layers.attention import (
    validate_native_attention_config,
    validate_native_layer_windows,
)

logger = logging.getLogger(__name__)
BRIDGE_PREFIX = "selected_prefix_bridge."


def resolve_dspine_head_topk(config, num_layers: int) -> int:
    """The refinement width K; fail loud unless the checkpoint states it unambiguously."""
    dflash_cfg = _get_dflash_config(config)
    found = {}
    second = getattr(config, "code_bridge_second_pass", None)
    if second is not None:
        second = dict(second)
        unknown = set(second) - {"mode", "first_exact", "head_topk"}
        if unknown:
            raise ValueError(f"Unknown code_bridge_second_pass keys: {sorted(unknown)}")
        if second.get("mode") != "serial_table":
            raise ValueError(
                f"DSPINE serves the serial last-write refinement only; the checkpoint config says code_bridge_second_pass.mode={second.get('mode')!r}"
            )
        if int(second.get("first_exact", -1)) != num_layers - 1:
            raise ValueError(
                f"serial_table redoes the LAST write only: code_bridge_second_pass.first_exact must be {num_layers - 1}, got {second.get('first_exact')!r}"
            )
        found["code_bridge_second_pass.head_topk"] = int(second["head_topk"])
    refine = dflash_cfg.get("selected_prefix_last_write_refine")
    if refine is not None:
        refine = dict(refine)
        if (
            set(refine) - {"kind", "version", "head_topk", "weight"}
            or not {"kind", "head_topk"} <= set(refine)
            or refine["kind"] != "exact_sender"
            or refine.get("version", 1) != 1
            or (not refine.get("weight", 1.0) >= 0)
        ):
            raise ValueError(
                f"DSPINE expects selected_prefix_last_write_refine = {{kind: 'exact_sender', version: 1, head_topk, weight >= 0}}, got {refine!r}"
            )
        found["dflash_config.selected_prefix_last_write_refine.head_topk"] = int(
            refine["head_topk"]
        )
    if not found:
        raise ValueError(
            "DSPINE checkpoint config names no refinement width: add code_bridge_second_pass {mode: serial_table, first_exact, head_topk} or dflash_config.selected_prefix_last_write_refine.head_topk"
        )
    values = set(found.values())
    if len(values) != 1:
        raise ValueError(f"DSPINE refinement widths disagree: {found}")
    k = values.pop()
    if k & k - 1 or not 2 <= k <= 64:
        raise ValueError(f"DSPINE head_topk must be a power of two in [2, 64], got {k}")
    return k


class DSpineCodeBridge(nn.Module):
    """Graph-capturable post-layer code writes with anchor-row (DFlash) semantics.  Weights load
    by their checkpoint names under ``selected_prefix_bridge.``; terminal-corrected bridges are
    refused. The bridge always uses its fused kernels."""

    def __init__(self, config):
        super().__init__()
        cfg = _get_dflash_config(config)
        if cfg.get("selected_prefix_terminal", True):
            raise ValueError("DSPINE supports terminal-disabled code bridges only")
        if cfg.get("selected_prefix_feature_write_mode") != "independent":
            raise ValueError("DSPINE requires independent learned writes")
        if cfg.get("selected_prefix_feature_writes") is not True:
            raise ValueError("DSPINE requires selected_prefix_feature_writes = true")
        if not cfg.get("draft_block_causal"):
            raise ValueError(
                "DSPINE requires causal draft attention (draft_block_causal)"
            )
        if cfg.get("selected_prefix_feature_reliability") is not None:
            raise ValueError("Reliability probes are not part of the DSPINE path")
        if cfg.get("selected_prefix_feature_next_token_rows", False):
            raise ValueError(
                "DSPINE's serial refinement is defined for anchor-row (DFlash) semantics; next-token rows are a DSpark layout"
            )
        bridge = cfg.get("selected_prefix_code_bridge")
        if not bridge:
            raise ValueError(
                "DSPINE draft config lacks dflash_config.selected_prefix_code_bridge"
            )
        self.hidden_size = int(config.hidden_size)
        self.code_size = int(bridge["code_size"])
        self.rank = int(bridge["rank"])
        self.num_layers = int(config.num_hidden_layers)
        self.block_size = int(config.block_size)
        self.eps = float(bridge["eps"])
        self.active_layers = frozenset(
            (int(i) - 1 for i in cfg["selected_prefix_feature_layers"])
        )
        if not self.active_layers <= set(range(self.num_layers)):
            raise ValueError(
                f"selected_prefix_feature_layers out of range: {sorted(self.active_layers)}"
            )
        if self.num_layers - 1 not in self.active_layers:
            raise ValueError(
                "DSPINE refines the last layer's write, so the last layer must be active"
            )
        self.terminal_residual_rank = cfg.get(
            "selected_prefix_terminal_residual_rank", 0
        )
        if (
            type(self.terminal_residual_rank) is not int
            or not 0 <= self.terminal_residual_rank <= self.rank
        ):
            raise ValueError(
                f"selected_prefix_terminal_residual_rank must be an int in [0, rank], got {self.terminal_residual_rank!r}"
            )
        if self.terminal_residual_rank and (
            cfg.get("selected_prefix_last_write_refine") is None
            or cfg.get("selected_prefix_feature_gt_training") is not True
        ):
            raise ValueError(
                "The terminal token residual needs the exact-code last-write refinement (selected_prefix_last_write_refine, selected_prefix_feature_gt_training = true)"
            )
        self.implementation = "fused"
        self.prev = nn.Linear(self.code_size, self.rank, bias=False)
        self.gate = nn.Linear(self.code_size, self.rank, bias=True)
        self.predecessor_gate = nn.Linear(self.code_size, self.rank, bias=False)
        self.out = nn.Linear(self.rank, self.code_size, bias=False)
        self.register_buffer(
            "codebook",
            torch.empty(config.vocab_size, self.code_size, dtype=torch.bfloat16),
        )
        self.register_buffer(
            "read_weight",
            torch.empty(self.code_size, self.hidden_size, dtype=torch.bfloat16),
        )
        self.register_buffer(
            "early_read_weights",
            torch.empty(
                self.num_layers - 1,
                self.code_size,
                self.hidden_size,
                dtype=torch.bfloat16,
            ),
        )
        self.feature_write_weights = nn.Parameter(
            torch.empty(
                self.num_layers, self.hidden_size, self.rank, dtype=torch.bfloat16
            ),
            requires_grad=False,
        )
        rt = self.terminal_residual_rank
        self.register_buffer(
            "terminal_residual_embedding",
            torch.empty(config.vocab_size, rt, dtype=torch.bfloat16) if rt else None,
        )
        self.register_buffer(
            "terminal_residual_projection",
            torch.empty(self.rank, rt, dtype=torch.bfloat16) if rt else None,
        )
        for name in ("independent_write_version", "terminal_disabled_version"):
            self.register_buffer(name, torch.zeros((), dtype=torch.int64))
        self.register_buffer("packed_weight", None, persistent=False)
        self._loaded = set()
        self._ready = False

    def load_weight(self, name: str, tensor: torch.Tensor) -> None:
        destinations = dict(self.named_parameters()) | dict(self.named_buffers())
        if name not in destinations or name == "packed_weight":
            raise ValueError(f"Unexpected DSPINE code-bridge weight: {name}")
        dest = destinations[name]
        if tuple(dest.shape) != tuple(tensor.shape):
            raise ValueError(
                f"DSPINE code-bridge shape mismatch: {name}: {tuple(tensor.shape)} vs {tuple(dest.shape)}"
            )
        dest.data.copy_(tensor)
        self._loaded.add(name)

    def begin_load(self) -> None:
        self._loaded.clear()
        self._ready = False

    def prepare(self) -> None:
        expected = (
            set(dict(self.named_parameters())) | set(dict(self.named_buffers()))
        ) - {"packed_weight"}
        missing = expected - self._loaded
        if missing:
            raise ValueError(f"Missing DSPINE code-bridge tensors: {sorted(missing)}")
        if self.out.weight.count_nonzero().item():
            raise ValueError(
                "Terminal-disabled checkpoint contains a nonzero terminal projection"
            )
        for name in ("independent_write_version", "terminal_disabled_version"):
            if getattr(self, name).item() != 1:
                raise ValueError(f"Unsupported DSPINE {name}")
        packed = torch.cat(
            (self.gate.weight, self.predecessor_gate.weight, self.prev.weight), dim=0
        ).contiguous()
        if self.packed_weight is None:
            self.packed_weight = packed
        else:
            self.packed_weight.copy_(packed)
        self._ready = True

    def layer_read_weight(self, layer: int) -> torch.Tensor:
        return (
            self.read_weight
            if layer == self.num_layers - 1
            else self.early_read_weights[layer]
        )

    @torch.no_grad()
    def fold_terminal_sender_table(
        self, table: torch.Tensor, chunk: int = 4096
    ) -> torch.Tensor:
        """Add the terminal token residual to the value half of a sender table in place:
        table[v, R:] = dtype(float(table[v, R:]) + float(dtype(E[v]) @ dtype(B)^T)), dtype = the
        table's (BF16 for the greedy TSV table).  The engine's fold_terminal_sender_table, same
        rounding points.  table [V, 2R] = [G_p e_v; A e_v]; a no-op without the residual."""
        if not self.terminal_residual_rank:
            return table
        (v, r) = (int(self.codebook.shape[0]), self.rank)
        if tuple(table.shape) != (v, 2 * r) or table.dtype not in (
            torch.bfloat16,
            torch.float32,
        ):
            raise ValueError(
                f"DSPINE terminal sender table has an unexpected shape or precision: {tuple(table.shape)} {table.dtype}"
            )
        weight = self.terminal_residual_projection.to(table.dtype)
        for start in range(0, v, chunk):
            end = min(start + chunk, v)
            residual = F.linear(
                self.terminal_residual_embedding[start:end].to(table.dtype), weight
            )
            table[start:end, r:] = (table[start:end, r:].float() + residual.float()).to(
                table.dtype
            )
        return table

    @torch.no_grad()
    def forward(
        self,
        hidden: torch.Tensor,
        residual: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        layer: int,
        next_norm: nn.Module,
        stash: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One write after ``layer``.  Returns (next_norm(out), out) where out is the updated
        residual stream; ``stash`` (a dict) receives the pre-write state raw / rms / packed."""
        if not self._ready:
            raise RuntimeError("DSPINE code-bridge weights were not prepared")
        if hidden.shape[0] % self.block_size:
            raise ValueError(
                "DSPINE code-bridge input must contain whole request blocks"
            )
        if hidden.dtype != torch.bfloat16:
            raise ValueError("The DSPINE code bridge requires BF16 activations")
        return self._forward_fused(hidden, residual, input_ids, layer, next_norm, stash)

    def _read_codes(self, hidden, residual, input_ids, layer):
        """Fused read: (raw [m,H] bf16, rms [m] fp32, codes [m,C] bf16)."""
        (m, h) = hidden.shape
        (c, t) = (self.code_size, self.block_size)
        raw = torch.empty_like(hidden)
        normalized = torch.empty_like(hidden)
        rms = torch.empty(m, device=hidden.device, dtype=torch.float32)
        kernels.prepare_read[m,](
            hidden,
            hidden if residual is None else residual,
            raw,
            normalized,
            rms,
            h,
            self.eps,
            residual is not None,
            triton.next_power_of_2(h),
            enable_fp_fusion=False,
        )
        projected = F.linear(normalized, self.layer_read_weight(layer))
        codes = torch.empty_like(projected)
        anchors = input_ids.view(-1, t)[:, 0].contiguous()
        kernels.normalize_codes[m,](
            projected,
            self.codebook,
            anchors,
            codes,
            c,
            t,
            True,
            self.eps,
            triton.next_power_of_2(c),
            enable_fp_fusion=False,
        )
        return (raw, rms, codes)

    def _forward_fused(self, hidden, residual, input_ids, layer, next_norm, stash):
        m = hidden.shape[0]
        (r, t, h) = (self.rank, self.block_size, self.hidden_size)
        (raw, rms, codes) = self._read_codes(hidden, residual, input_ids, layer)
        packed = F.linear(codes, self.packed_weight)
        if stash is not None:
            stash.update(raw=raw, rms=rms, packed=packed)
        message = torch.empty((m, r), device=hidden.device, dtype=hidden.dtype)
        kernels.feature_message[m,](
            packed,
            self.gate.bias,
            message,
            r,
            t,
            triton.next_power_of_2(r),
            enable_fp_fusion=False,
        )
        delta = F.linear(message, self.feature_write_weights[layer])
        valid = torch.ones(m, device=raw.device, dtype=torch.bool)
        output = torch.empty_like(raw)
        normed = torch.empty_like(raw)
        kernels.feature_finish[m,](
            raw,
            delta,
            rms,
            valid,
            next_norm.weight,
            output,
            normed,
            h,
            t,
            True,
            True,
            next_norm.variance_epsilon,
            triton.next_power_of_2(h),
            enable_fp_fusion=False,
        )
        return (normed, output)


class DSpineDecoderLayer(DFlashDecoderLayer):
    """DFlash layer whose input RMSNorm may already have been applied by a fused code write."""

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        input_normed: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hidden_states.numel() == 0:
            if residual is None:
                residual = hidden_states
            return (hidden_states, residual)
        if input_normed:
            if residual is None:
                raise RuntimeError("A pre-normed DSPINE layer input needs its residual")
        elif residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            (hidden_states, residual) = self.input_layernorm(hidden_states, residual)
        attn_out = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        (hidden_states, residual) = self.post_attention_layernorm(attn_out, residual)
        hidden_states = self.mlp(hidden_states)
        return (hidden_states, residual)


class DSpineDraftModel(DFlashDraftModel):
    """DFlash draft + code bridge; the refinement tail runs in DSpineDraftSampler."""

    decoder_layer_cls = DSpineDecoderLayer

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        if quant_config is not None:
            raise ValueError("The DSPINE draft does not support quantized weights")
        if _get_dflash_config(config).get("conv_kernel_size"):
            raise ValueError("DSpine checkpoints do not use grouped draft convolutions")
        if _get_dflash_config(config).get("draft_block_causal"):
            text_config = config.get_text_config()
            if getattr(text_config, "is_causal", None) is None:
                text_config.is_causal = True
        self.native_swa_window = validate_native_attention_config(config)
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        cfg = _get_dflash_config(config)
        unsupported = [
            key
            for key in (
                "selector_rank",
                "selector_top_k",
                "projector_type",
                "attention_sink_bias",
            )
            if cfg.get(key)
        ]
        if cfg.get("attention_value_scale") is not None:
            unsupported.append("attention_value_scale")
        if getattr(config, "has_embed_tokens", False):
            unsupported.append("has_embed_tokens")
        if float(getattr(config, "partial_rotary_factor", None) or 1.0) != 1.0:
            unsupported.append("partial_rotary_factor")
        if unsupported:
            raise ValueError(
                f"DSPINE drafts are plain DFlash layers with a code bridge; unsupported draft config keys: {unsupported}"
            )
        if getattr(self, "is_nemotron_35_draft", False):
            raise ValueError("Nemotron 3.5 draft layouts are not DSPINE drafts")
        validate_native_layer_windows(
            self.layers, self.native_swa_window, decoder_type=AttentionType.DECODER
        )
        self.selected_prefix_bridge = DSpineCodeBridge(config)
        if self.selected_prefix_bridge.block_size != int(self.block_size):
            raise ValueError("DSPINE bridge and draft block sizes differ")
        self.trained_block_size = int(self.block_size)
        self.head_topk = resolve_dspine_head_topk(config, len(self.layers))
        self.trained_head_topk = self.head_topk
        self.last_write_stash: Optional[dict] = None
        self.dspine_sampler = None

    def set_block_size(self, block_size: int) -> None:
        """Serve with ``block_size`` rows per block (module docstring): 2 <= block_size <= the trained
        block size.  The block size reaches every part that locates rows inside a block -- the code
        bridge (anchor rows, predecessor messages) and the convolutions (taps stay inside a block); the
        DSpine sampler is built with the worker's block size.  Call before any draft forward is
        captured: the size is read at call time and becomes a constant of a captured graph."""
        t = int(block_size)
        if not 2 <= t <= self.trained_block_size:
            raise ValueError(
                f"DSPINE serves block sizes 2..{self.trained_block_size} (the trained block size) for this draft; got {t}: rows past the trained block were never trained"
            )
        for layer in self.layers:
            for conv in (layer.attention_conv, layer.mlp_conv):
                if conv is not None and conv.taps > t:
                    raise ValueError(
                        f"DSPINE conv_kernel_size={conv.taps} exceeds the served block size {t}"
                    )
        self.block_size = t
        self.selected_prefix_bridge.block_size = t
        for layer in self.layers:
            for conv in (layer.attention_conv, layer.mlp_conv):
                if conv is not None:
                    conv.block_size = t

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        if input_embeds is None:
            raise ValueError(
                "DSpineDraftModel requires `input_embeds` (the target embedding)"
            )
        bridge = self.selected_prefix_bridge
        last = len(self.layers) - 1
        hidden_states = input_embeds
        residual: Optional[torch.Tensor] = None
        input_normed = False
        self.last_write_stash = None
        for index, layer in enumerate(self.layers):
            (hidden_states, residual) = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
                input_normed=input_normed,
            )
            input_normed = False
            if index in bridge.active_layers and hidden_states.numel():
                next_norm = (
                    self.layers[index + 1].input_layernorm
                    if index < last
                    else self.norm
                )
                stash = {} if index == last else None
                (hidden_states, residual) = bridge(
                    hidden_states, residual, input_ids, index, next_norm, stash
                )
                if stash is not None:
                    self.last_write_stash = stash
                input_normed = True
        if hidden_states.numel() != 0 and (not input_normed):
            raise RuntimeError(
                "DSPINE's last layer must end with a (norm-fused) code write"
            )
        return LogitsProcessorOutput(
            next_token_logits=None, hidden_states=hidden_states
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        bridge = self.selected_prefix_bridge
        bridge.begin_load()
        dflash_params = [
            (name, p)
            for (name, p) in self.named_parameters()
            if not name.startswith(BRIDGE_PREFIX) and p.dtype.is_floating_point
        ]
        with torch.no_grad():
            for _, p in dflash_params:
                p.fill_(float("nan"))
        causal_seen = []
        refine_version_seen = []

        def dflash_weights():
            for name, tensor in weights:
                short = name.removeprefix("model.")
                if short.startswith(BRIDGE_PREFIX):
                    bridge.load_weight(short[len(BRIDGE_PREFIX) :], tensor)
                    continue
                if short == "draft_block_causal_version":
                    if int(tensor.item()) != 1:
                        raise ValueError("Unsupported causal draft checkpoint version")
                    causal_seen.append(True)
                    continue
                if short == "last_write_refine_version":
                    if int(tensor.item()) != 1:
                        raise ValueError(
                            "Unsupported last-write refinement checkpoint version"
                        )
                    refine_version_seen.append(True)
                    continue
                if short.startswith(
                    ("prefix_gru.", "embed_proj.", "candidate_selector.")
                ):
                    raise ValueError(
                        f"DSPINE drafts carry no Domino / DFlash2 selector weights ({name})"
                    )
                if _loader_param_name(short) not in param_names:
                    raise ValueError(
                        f"DSPINE draft checkpoint tensor {name!r} matches no parameter of this draft (check the checkpoint config)"
                    )
                yield (name, tensor)

        param_names = set(dict(self.named_parameters()))
        super().load_weights(dflash_weights())
        unloaded = [name for (name, p) in dflash_params if torch.isnan(p).any().item()]
        if unloaded:
            raise ValueError(f"DSPINE checkpoint left parameters unloaded: {unloaded}")
        bridge.prepare()
        conv = self.layers[0].attention_conv
        logger.info(
            "Loaded DSPINE draft: layers=%d bridge=%s active_layers=%s code_size=%d rank=%d head_topk=%d terminal_residual_rank=%d conv=%s causal_version_tensor=%s refine_version_tensor=%s",
            len(self.layers),
            bridge.implementation,
            sorted((i + 1 for i in bridge.active_layers)),
            bridge.code_size,
            bridge.rank,
            self.head_topk,
            bridge.terminal_residual_rank,
            f"taps{conv.taps}/group{conv.group_size}" if conv is not None else "none",
            bool(causal_seen),
            bool(refine_version_seen),
        )


_STACKED_NAMES = (
    ("q_proj", "qkv_proj"),
    ("k_proj", "qkv_proj"),
    ("v_proj", "qkv_proj"),
    ("gate_proj", "gate_up_proj"),
    ("up_proj", "gate_up_proj"),
)


def _loader_param_name(short: str) -> str:
    """The parameter a (``model.``-stripped) checkpoint name lands in under SGLang 0.5.22's
    DFlashDraftModel.load_weights."""
    for weight_name, param_name in _STACKED_NAMES:
        if f".{weight_name}." in short:
            return short.replace(weight_name, param_name)
    return short


EntryClass = [DSpineDraftModel]
