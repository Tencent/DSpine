from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


@dataclass(frozen=True)
class BridgeConfig:
    code_size: int
    rank: int
    eps: float
    kappa: float | None = None


@dataclass(frozen=True)
class DraftConfig:
    """Parsed DSpine draft config (HF ``config.json``)."""

    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    block_size: int
    mask_token_id: int
    target_layer_ids: tuple[int, ...]
    num_target_layers: int
    bridge: BridgeConfig
    feature_layers: tuple[int, ...]  # 0-based draft layer ids
    head_topk: int
    trained_block_size: int
    architectures: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def concat_size(self) -> int:
        return len(self.target_layer_ids) * self.hidden_size

    def validate(self) -> None:
        _require(self.hidden_size > 0, "hidden_size must be positive")
        _require(self.num_hidden_layers > 0, "num_hidden_layers must be positive")
        _require(self.intermediate_size > 0, "intermediate_size must be positive")
        _require(self.num_attention_heads > 0, "num_attention_heads must be positive")
        _require(self.num_key_value_heads > 0, "num_key_value_heads must be positive")
        _require(
            self.num_attention_heads % self.num_key_value_heads == 0,
            "num_attention_heads must be divisible by num_key_value_heads",
        )
        _require(self.head_dim > 0, "head_dim must be positive")
        _require(self.vocab_size > 0, "vocab_size must be positive")
        _require(self.rms_norm_eps > 0, "rms_norm_eps must be positive")
        _require(self.rope_theta > 0, "rope_theta must be positive")
        _require(self.max_position_embeddings > 0, "max_position_embeddings must be positive")
        _require(self.block_size >= 2, "block_size must be >= 2")
        _require(
            0 <= self.mask_token_id < self.vocab_size,
            f"mask_token_id {self.mask_token_id} outside vocab {self.vocab_size}",
        )
        _require(self.num_target_layers > 0, "num_target_layers must be positive")
        _require(len(self.target_layer_ids) > 0, "target_layer_ids must not be empty")
        _require(
            len(set(self.target_layer_ids)) == len(self.target_layer_ids),
            f"duplicate target_layer_ids: {self.target_layer_ids}",
        )
        for lid in self.target_layer_ids:
            _require(
                0 <= lid < self.num_target_layers,
                f"target_layer_ids {lid} outside [0,{self.num_target_layers})",
            )
        _require(self.bridge.code_size > 0, "bridge code_size must be positive")
        _require(self.bridge.rank > 0, "bridge rank must be positive")
        _require(self.bridge.eps > 0, "bridge eps must be positive")
        _require(
            self.feature_layers == tuple(range(self.num_hidden_layers)),
            "DSpine MLX port requires every draft layer to carry a bridge write "
            f"(got {self.feature_layers})",
        )
        k = self.head_topk
        _require(
            k & (k - 1) == 0 and 2 <= k <= 64,
            f"head_topk must be a power of two in [2,64], got {k}",
        )
        if self.architectures:
            _require(
                any(a in ("DFlashDraftModel", "DSpineDraftModel") for a in self.architectures),
                f"unsupported draft architectures: {self.architectures}",
            )
        if self.raw.get("model_type") is not None:
            _require(
                self.raw["model_type"] == "qwen3",
                f"unsupported draft model_type {self.raw['model_type']!r}",
            )
        if self.raw.get("attention_bias") is not None:
            _require(not self.raw["attention_bias"], "draft attention_bias must be false")
        if self.raw.get("hidden_act") is not None:
            _require(
                self.raw["hidden_act"] == "silu",
                f"unsupported draft hidden_act {self.raw['hidden_act']!r}",
            )
        if self.raw.get("tie_word_embeddings") is not None:
            _require(
                not self.raw["tie_word_embeddings"],
                "draft tie_word_embeddings must be false (it shares the target head)",
            )
        if self.raw.get("draft_vocab_size") is not None:
            _require(
                int(self.raw["draft_vocab_size"]) == self.vocab_size,
                "draft_vocab_size != vocab_size",
            )

    def validate_served_block_size(self, block_size: int) -> None:
        t = int(block_size)
        _require(
            2 <= t <= self.trained_block_size,
            f"DSpine serves block sizes 2..{self.trained_block_size}; got {t}",
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any], path: str | Path | None = None) -> "DraftConfig":
        where = f" ({path})" if path is not None else ""
        dflash = raw.get("dflash_config")
        _require(isinstance(dflash, dict), f"missing dflash_config{where}")

        # The DFlash2 / Domino / DSpark heads a DSpine draft does not have.
        for key in ("selector_rank", "selector_top_k", "projector_type", "attention_sink_bias"):
            _require(not dflash.get(key), f"unsupported DSpine draft key {key}{where}")
        _require(
            dflash.get("attention_value_scale") is None,
            f"unsupported attention_value_scale{where}",
        )
        _require(
            dflash.get("selected_prefix_feature_reliability") is None,
            f"unsupported selected_prefix_feature_reliability{where}",
        )
        _require(
            not dflash.get("selected_prefix_feature_next_token_rows", False),
            f"unsupported selected_prefix_feature_next_token_rows{where}",
        )
        taps = dflash.get("conv_kernel_size", 0) or 0
        group = dflash.get("conv_group_size", 0) or 0
        _require(
            not taps and not group,
            f"the MLX DSpine port does not implement the grouped convolution{where}",
        )

        _require(
            dflash.get("selected_prefix_terminal") is False,
            f"DSpine MLX requires selected_prefix_terminal = false{where}",
        )
        _require(
            dflash.get("selected_prefix_decode_mode") == "serial",
            f"DSpine MLX requires selected_prefix_decode_mode = serial{where}",
        )
        _require(
            dflash.get("selected_prefix_feature_writes") is True,
            f"DSpine MLX requires selected_prefix_feature_writes = true{where}",
        )
        _require(
            dflash.get("selected_prefix_feature_write_mode") == "independent",
            f"DSpine MLX requires independent writes{where}",
        )
        _require(
            dflash.get("draft_block_causal") is True,
            f"DSpine MLX requires draft_block_causal = true{where}",
        )

        bridge_raw = dflash.get("selected_prefix_code_bridge")
        _require(isinstance(bridge_raw, dict), f"missing selected_prefix_code_bridge{where}")
        bridge = BridgeConfig(
            code_size=int(bridge_raw["code_size"]),
            rank=int(bridge_raw["rank"]),
            eps=float(bridge_raw["eps"]),
            kappa=bridge_raw.get("kappa"),
        )

        num_layers = int(raw["num_hidden_layers"])
        feature_raw = dflash.get("selected_prefix_feature_layers")
        _require(isinstance(feature_raw, list), f"missing selected_prefix_feature_layers{where}")
        feature_layers = tuple(int(x) - 1 for x in feature_raw)
        _require(
            all(0 <= x < num_layers for x in feature_layers),
            f"selected_prefix_feature_layers out of range{where}",
        )

        refine = dflash.get("selected_prefix_last_write_refine")
        second = raw.get("code_bridge_second_pass")
        found: list[int] = []
        if refine is not None:
            _require(isinstance(refine, dict), f"bad selected_prefix_last_write_refine{where}")
            _require(
                set(refine) == {"kind", "version", "head_topk", "weight"},
                f"selected_prefix_last_write_refine has unexpected keys{where}: {sorted(refine)}",
            )
            _require(
                refine["kind"] == "exact_sender",
                f"DSpine requires kind = exact_sender{where}",
            )
            _require(int(refine["version"]) == 1, f"unsupported refine version{where}")
            _require(float(refine["weight"]) >= 0, f"refine weight must be >= 0{where}")
            found.append(int(refine["head_topk"]))
        if second is not None:
            _require(isinstance(second, dict), f"bad code_bridge_second_pass{where}")
            _require(
                set(second) <= {"mode", "first_exact", "head_topk"},
                f"code_bridge_second_pass has unexpected keys{where}",
            )
            _require(
                second.get("mode") == "serial_table",
                f"DSpine serves serial_table only{where}",
            )
            _require(
                int(second.get("first_exact", -1)) == num_layers - 1,
                f"serial_table must redo the last layer only{where}",
            )
            found.append(int(second["head_topk"]))
        _require(bool(found), f"checkpoint names no refinement width{where}")
        _require(len(set(found)) == 1, f"refinement widths disagree: {found}{where}")
        head_topk = found[0]

        layer_types = raw.get("layer_types")
        if layer_types is not None:
            _require(
                all(t == "full_attention" for t in layer_types),
                f"DSpine drafts use full attention only{where}",
            )
            _require(
                len(layer_types) == num_layers,
                f"layer_types length != num_hidden_layers{where}",
            )
        _require(
            not raw.get("use_sliding_window", False),
            f"DSpine drafts use no sliding window{where}",
        )
        _require(
            raw.get("sliding_window") is None,
            f"DSpine drafts use no sliding window{where}",
        )

        rope_theta = raw.get("rope_theta")
        rope_params = raw.get("rope_parameters")
        if rope_theta is None and isinstance(rope_params, dict):
            rope_theta = rope_params.get("rope_theta")
        _require(rope_theta is not None, f"missing rope_theta{where}")
        if isinstance(rope_params, dict):
            rope_type = rope_params.get("rope_type", rope_params.get("type", "default"))
            _require(
                rope_type == "default",
                f"DSpine MLX supports default RoPE only, got {rope_type!r}{where}",
            )
        _require(
            raw.get("rope_scaling") is None,
            f"DSpine MLX supports default RoPE only{where}",
        )

        block_size = int(raw["block_size"])
        cfg = cls(
            hidden_size=int(raw["hidden_size"]),
            num_hidden_layers=num_layers,
            intermediate_size=int(raw["intermediate_size"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            num_key_value_heads=int(raw["num_key_value_heads"]),
            head_dim=int(raw["head_dim"]),
            vocab_size=int(raw["vocab_size"]),
            rms_norm_eps=float(raw["rms_norm_eps"]),
            rope_theta=float(rope_theta),
            max_position_embeddings=int(raw["max_position_embeddings"]),
            block_size=block_size,
            mask_token_id=int(dflash["mask_token_id"]),
            target_layer_ids=tuple(int(x) for x in dflash["target_layer_ids"]),
            num_target_layers=int(raw["num_target_layers"]),
            bridge=bridge,
            feature_layers=feature_layers,
            head_topk=head_topk,
            trained_block_size=block_size,
            architectures=tuple(raw.get("architectures", ())),
            raw=raw,
        )
        cfg.validate()
        return cfg


def resolve_draft_path(path: str | Path, revision: str | None = None) -> Path:
    """Return a local draft path, downloading a Hugging Face repo id if needed."""
    local = Path(path).expanduser()
    if local.exists():
        return local
    from huggingface_hub import snapshot_download

    try:
        return Path(
            snapshot_download(
                str(path),
                revision=revision,
                allow_patterns=["*.json", "*.safetensors"],
            )
        )
    except Exception as e:
        raise FileNotFoundError(
            f"DSpine draft {str(path)!r} is neither a local path nor a "
            f"downloadable Hugging Face repo: {e}"
        ) from e


def load_draft_config(path: str | Path) -> DraftConfig:
    """Read ``<draft>/config.json`` and validate the DSpine inference contract."""
    root = resolve_draft_path(path)
    cfg_path = root / "config.json" if root.is_dir() else root
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return DraftConfig.from_dict(raw, cfg_path)
