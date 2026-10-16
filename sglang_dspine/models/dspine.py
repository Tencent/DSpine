"""DSpine draft model with per-layer code bridges and serial last-write refinement.

The block's first row is the last committed token (the anchor). Bridge writes
condition each subsequent row on its predecessor; the refinement tail runs in
``DSpineDraftSampler``.
"""

from __future__ import annotations

import logging
import math
import os
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

from sglang_dspine import kernels

logger = logging.getLogger(__name__)

BRIDGE_PREFIX = "selected_prefix_bridge."


def resolve_dspine_head_topk(config, num_layers: int) -> int:
    """Resolve the width K for serial last-write refinement.

    Read ``code_bridge_second_pass`` (mode ``serial_table``, ``first_exact`` is the
    last layer) and/or ``dflash_config.selected_prefix_last_write_refine`` (kind
    ``exact_sender``). Both sources must agree when present.
    """
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
                "DSPINE serves the serial last-write refinement only; the checkpoint config says "
                f"code_bridge_second_pass.mode={second.get('mode')!r}"
            )
        if int(second.get("first_exact", -1)) != num_layers - 1:
            raise ValueError(
                "serial_table redoes the LAST write only: code_bridge_second_pass.first_exact "
                f"must be {num_layers - 1}, got {second.get('first_exact')!r}"
            )
        found["code_bridge_second_pass.head_topk"] = int(second["head_topk"])
    refine = dflash_cfg.get("selected_prefix_last_write_refine")
    if refine is not None:
        if refine.get("kind") != "exact_sender":
            raise ValueError(
                "DSPINE expects selected_prefix_last_write_refine.kind == 'exact_sender', "
                f"got {refine.get('kind')!r}"
            )
        found["dflash_config.selected_prefix_last_write_refine.head_topk"] = int(
            refine["head_topk"]
        )
    if not found:
        raise ValueError(
            "DSPINE checkpoint config names no refinement width: add "
            "code_bridge_second_pass {mode: serial_table, first_exact, head_topk} or "
            "dflash_config.selected_prefix_last_write_refine.head_topk"
        )
    values = set(found.values())
    if len(values) != 1:
        raise ValueError(f"DSPINE refinement widths disagree: {found}")
    k = values.pop()
    if k & (k - 1) or not 2 <= k <= 64:
        raise ValueError(f"DSPINE head_topk must be a power of two in [2, 64], got {k}")
    return k


class DSpineCodeBridge(nn.Module):
    """Graph-capturable post-layer writes with DFlash anchor-row semantics.

    ``dflash_config.selected_prefix_code_bridge`` specifies ``code_size``, ``rank``,
    and ``eps``. ``selected_prefix_feature_layers`` is 1-based and includes the last
    layer; ``selected_prefix_feature_writes`` and ``draft_block_causal`` are true,
    ``selected_prefix_feature_write_mode`` is ``independent``, and
    ``selected_prefix_terminal`` is false.

    Each write fuses the next layer's input RMSNorm (or final norm).
    ``SGLANG_DSPINE_BRIDGE_IMPL=eager`` uses plain torch operations.
    """

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
                "DSPINE's serial refinement is defined for anchor-row (DFlash) semantics; "
                "next-token rows are a DSpark layout"
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
            int(i) - 1 for i in cfg["selected_prefix_feature_layers"]
        )
        if not self.active_layers <= set(range(self.num_layers)):
            raise ValueError(
                f"selected_prefix_feature_layers out of range: {sorted(self.active_layers)}"
            )
        if self.num_layers - 1 not in self.active_layers:
            raise ValueError(
                "DSPINE refines the last layer's write, so the last layer must be active"
            )
        implementation = os.environ.get("SGLANG_DSPINE_BRIDGE_IMPL", "fused")
        if implementation not in ("fused", "eager"):
            raise ValueError(f"Unknown DSPINE bridge implementation {implementation!r}")
        self.implementation = implementation
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
                f"DSPINE code-bridge shape mismatch: {name}: {tuple(tensor.shape)} vs "
                f"{tuple(dest.shape)}"
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
        # Preserve graph-captured storage during same-shape weight reloads.
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
        if self.implementation == "fused":
            return self._forward_fused(
                hidden, residual, input_ids, layer, next_norm, stash
            )
        return self._forward_eager(hidden, residual, input_ids, layer, next_norm, stash)

    def _read_codes(self, hidden, residual, input_ids, layer):
        """Fused read: (raw [m,H] bf16, rms [m] fp32, codes [m,C] bf16)."""
        m, h = hidden.shape
        c, t = self.code_size, self.block_size
        raw = torch.empty_like(hidden)
        normalized = torch.empty_like(hidden)
        rms = torch.empty(m, device=hidden.device, dtype=torch.float32)
        kernels.prepare_read[(m,)](
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
        kernels.normalize_codes[(m,)](
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
        return raw, rms, codes

    def _forward_fused(self, hidden, residual, input_ids, layer, next_norm, stash):
        m = hidden.shape[0]
        r, t, h = self.rank, self.block_size, self.hidden_size
        raw, rms, codes = self._read_codes(hidden, residual, input_ids, layer)
        packed = F.linear(codes, self.packed_weight)
        if stash is not None:
            stash.update(raw=raw, rms=rms, packed=packed)
        message = torch.empty((m, r), device=hidden.device, dtype=hidden.dtype)
        kernels.feature_message[(m,)](
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
        kernels.feature_finish[(m,)](
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
        return normed, output

    def _forward_eager(self, hidden, residual, input_ids, layer, next_norm, stash):
        """Plain-torch version of the fused path with the same rounding points (the eager A/B
        path, and the reference the fused kernels are tested against)."""
        t, r = self.block_size, self.rank
        raw = hidden if residual is None else hidden + residual
        f = raw.float()
        rms = f.square().mean(-1).clamp_min(torch.finfo(torch.float32).tiny).sqrt()
        normalized = (f / (rms[:, None] + self.eps)).to(raw.dtype)
        projected = F.linear(normalized, self.layer_read_weight(layer)).float()
        codes = (
            math.sqrt(self.code_size)
            * projected
            / (projected.norm(dim=-1, keepdim=True) + self.eps)
        )
        anchors = input_ids.view(-1, t)[:, 0]
        anchor_codes = (
            math.sqrt(self.code_size) * F.embedding(anchors, self.codebook).float()
        )
        codes = codes.view(-1, t, self.code_size)
        codes = torch.cat((anchor_codes[:, None, :], codes[:, 1:]), dim=1).view(
            -1, self.code_size
        )
        packed = F.linear(codes.to(raw.dtype), self.packed_weight)
        if stash is not None:
            stash.update(raw=raw, rms=rms, packed=packed)
        q, s, v = packed.float().view(-1, t, 3 * r).split(r, dim=-1)
        gate = torch.sigmoid(q[:, 1:] + s[:, :-1] + self.gate.bias.float())
        message = torch.cat((torch.zeros_like(q[:, :1]), gate * v[:, :-1]), dim=1).to(
            raw.dtype
        )
        delta = F.linear(message.view(-1, r), self.feature_write_weights[layer]).float()
        not_anchor = (torch.arange(raw.shape[0], device=raw.device) % t != 0).float()[
            :, None
        ]
        out = (f + (rms[:, None] * delta) * not_anchor).to(raw.dtype)
        x = out.float()
        y = (
            x
            * torch.rsqrt(
                x.square().mean(-1, keepdim=True) + next_norm.variance_epsilon
            )
        ).to(raw.dtype)
        normed = (y.float() * next_norm.weight.float()).to(raw.dtype)
        return normed, out


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
            return hidden_states, residual
        if input_normed:
            if residual is None:
                raise RuntimeError("A pre-normed DSPINE layer input needs its residual")
        elif residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        attn_out = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states, residual = self.post_attention_layernorm(attn_out, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class DSpineDraftModel(DFlashDraftModel):
    """DFlash draft with a code bridge.

    Accepts ``DSpineDraftModel`` and bridge-bearing ``DFlashDraftModel`` checkpoint
    architectures, mapped by the plugin under DSPINE.
    """

    decoder_layer_cls = DSpineDecoderLayer

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        if quant_config is not None:
            raise ValueError("The DSPINE draft does not support quantized weights")
        # SGLang reads is_causal while constructing attention layers. Infer it from
        # draft_block_causal when missing or None. Preserve explicit False so the
        # causal-attention check below rejects it.
        if _get_dflash_config(config).get("draft_block_causal"):
            text_config = config.get_text_config()
            if getattr(text_config, "is_causal", None) is None:
                text_config.is_causal = True
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        dc = self.draft_config
        if dc.conv_kernel_size or dc.is_domino or dc.selector_top_k:
            raise ValueError(
                "DSPINE drafts carry neither DFlash2 convolutions/selector nor a Domino projector"
            )
        if self.is_nemotron_35_draft:
            raise ValueError("Nemotron 3.5 draft layouts are not DSPINE drafts")
        for layer in self.layers:
            if layer.self_attn.attn.attn_type != AttentionType.DECODER:
                raise ValueError(
                    "DSPINE requires causal draft attention in every layer (config is_causal = "
                    "true with layer_types full_attention)"
                )
            if layer.self_attn.sliding_window_size != -1:
                raise ValueError("DSPINE drafts use full attention layers")
        self.selected_prefix_bridge = DSpineCodeBridge(config)
        if self.selected_prefix_bridge.block_size != int(self.block_size):
            raise ValueError("DSPINE bridge and draft block sizes differ")
        self.head_topk = resolve_dspine_head_topk(config, len(self.layers))
        self.last_write_stash: Optional[dict] = None
        # The refinement tail (DSpineDraftSampler), registered by DSpineWorkerV2; the draft graph
        # runner (DSpineDraftCudaGraphRunner) finds it here after each replay.
        self.dspine_sampler = None

    def set_block_size(self, block_size: int) -> None:
        if int(block_size) != self.selected_prefix_bridge.block_size:
            raise ValueError(
                "DSPINE's code bridge is trained for block size "
                f"{self.selected_prefix_bridge.block_size}; --speculative-num-draft-tokens="
                f"{block_size} cannot be served"
            )
        super().set_block_size(block_size)

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
        """Return final-normed block states [bs*T, H] for the target LM head.

        Row 0 is the anchor; rows 1..T-1 provide pass-1 logits. ``last_write_stash``
        retains the last write's pre-write raw [bs*T, H] BF16, rms [bs*T] FP32, and
        packed [bs*T, 3R] BF16 state for ``DSpineDraftSampler``.
        """
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
            hidden_states, residual = layer(
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
                hidden_states, residual = bridge(
                    hidden_states, residual, input_ids, index, next_norm, stash
                )
                if stash is not None:
                    self.last_write_stash = stash
                input_normed = True
        if hidden_states.numel() != 0 and not input_normed:
            raise RuntimeError(
                "DSPINE's last layer must end with a (norm-fused) code write"
            )
        return LogitsProcessorOutput(
            next_token_logits=None, hidden_states=hidden_states
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load DFlash parameters and ``selected_prefix_bridge.*`` bridge tensors.

        The optional ``draft_block_causal_version`` tensor must be version 1.
        """
        bridge = self.selected_prefix_bridge
        bridge.begin_load()
        # Every non-bridge parameter must come from the checkpoint: poison them first and demand
        # that loading overwrote every element (the DFlash loader skips unknown names silently).
        dflash_params = [
            (name, p)
            for name, p in self.named_parameters()
            if not name.startswith(BRIDGE_PREFIX) and p.dtype.is_floating_point
        ]
        with torch.no_grad():
            for _, p in dflash_params:
                p.fill_(float("nan"))
        causal_seen = []

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
                yield name, tensor

        super().load_weights(dflash_weights())
        unloaded = [name for name, p in dflash_params if torch.isnan(p).any().item()]
        if unloaded:
            raise ValueError(f"DSPINE checkpoint left parameters unloaded: {unloaded}")
        bridge.prepare()
        logger.info(
            "Loaded DSPINE draft: layers=%d bridge=%s active_layers=%s code_size=%d rank=%d "
            "head_topk=%d causal_version_tensor=%s",
            len(self.layers),
            bridge.implementation,
            sorted(i + 1 for i in bridge.active_layers),
            bridge.code_size,
            bridge.rank,
            self.head_topk,
            bool(causal_seen),
        )


EntryClass = [DSpineDraftModel]
