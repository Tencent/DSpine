"""Measured verification widths and token budgets, indexed by actual batch size."""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SpsPolicy:
    # Missing batch sizes verify the full block; measured budgets are not
    # extrapolated to another batch size.
    verify_widths: dict[int, int] = field(default_factory=dict)
    token_budgets: dict[int, int] = field(default_factory=dict)

    def __post_init__(self):
        for values, name in (
            (self.verify_widths, "verify_widths"),
            (self.token_budgets, "token_budgets"),
        ):
            if not isinstance(values, dict):
                raise ValueError(f"SPS {name} must map positive batch sizes to integers")
            for bs, value in values.items():
                if type(bs) is not int or bs < 1 or type(value) is not int:
                    raise ValueError(f"SPS {name} requires positive integer batch sizes")
                lower, upper = (1, 16) if name == "verify_widths" else (bs, bs * 16)
                if not lower <= value <= upper:
                    raise ValueError(f"SPS {name}[{bs}] must be in [{lower}, {upper}]")
        if self.verify_widths.keys() & self.token_budgets.keys():
            raise ValueError("SPS batch sizes must use either a width or a token budget")

    def width(self, bs, full=16):
        return min(full, self.verify_widths.get(bs, full))

    def budget(self, bs, full=16):
        return min(bs * full, self.token_budgets.get(bs, bs * full))

    def to_dict(self):
        return {
            "verify_widths": {str(bs): w for bs, w in sorted(self.verify_widths.items())},
            "token_budgets": {str(bs): n for bs, n in sorted(self.token_budgets.items())},
        }

    @classmethod
    def load(cls, path, hardware, execution=None):
        return cls.load_with_modes(path, hardware, execution)[0]

    @classmethod
    def load_with_modes(cls, path, hardware, execution=None):
        """Load a fallback and optional mode overrides from the same profile."""
        if not path:
            return cls(), {}
        data = json.loads(Path(path).read_text())
        if data.get("format") not in ("dspine-sps-v1", "dspine-sps-v2"):
            raise ValueError("Expected a DSpine SPS profile from scripts/tune_sps.py")
        if data.get("hardware") != hardware:
            raise ValueError(
                "SPS profile hardware differs; rerun scripts/tune_sps.py on this GPU"
            )
        if execution is not None:
            recorded = data.get("execution")
            if recorded is None:
                if execution["tp_size"] != 1 or execution["draft_tp_size"] != 1:
                    raise ValueError(
                        "SPS profile has no TP layout; retune for this deployment"
                    )
                if execution.get("block_size", 16) != 16:
                    raise ValueError(
                        "SPS profile execution layout differs; retune for this deployment"
                    )
            else:
                for name in (
                    "tp_size",
                    "draft_tp_size",
                    "attention_backend",
                    "draft_attention_backend",
                ):
                    expected, actual = execution.get(name), recorded.get(name)
                    if (
                        name.endswith("tp_size")
                        or expected is not None
                        and actual is not None
                    ):
                        if expected != actual:
                            raise ValueError(
                                "SPS profile execution layout differs; retune for this deployment"
                            )
                recorded_block = recorded.get("block_size", 16)
                expected_block = execution.get("block_size", 16)
                if (
                    type(recorded_block) is not int
                    or not 2 <= recorded_block <= 16
                    or recorded_block != expected_block
                ):
                    raise ValueError(
                        "SPS profile execution layout differs; retune for this deployment"
                    )
        policy = data["policy"]
        if data["format"] == "dspine-sps-v1":
            widths, budget = policy["uniform_widths"], policy["batch8_tokens"]
            if (
                len(widths) != 7
                or any(type(w) is not int or not 2 <= w <= 16 for w in widths)
                or type(budget) is not int
                or budget not in range(16, 129, 16)
            ):
                raise ValueError("Invalid legacy DSpine SPS profile")
            default, mode_policies = cls(dict(enumerate(widths, 1)), {8: budget}), {}
        else:
            modes = data.get("mode_policies", {})
            if not isinstance(modes, dict) or set(modes) - {"greedy", "sampled"}:
                raise ValueError("SPS mode_policies only supports greedy and sampled")
            default = cls.from_dict(policy)
            mode_policies = {
                mode: cls.from_dict(value) for mode, value in modes.items()
            }
        block = (
            execution.get("block_size", 16)
            if execution is not None
            else (data.get("execution") or {}).get("block_size", 16)
        )
        if type(block) is not int or not 2 <= block <= 16:
            raise ValueError("SPS profile block_size must be an integer in 2..16")
        for item in (default, *mode_policies.values()):
            if any(w > block for w in item.verify_widths.values()) or any(
                budget > bs * block for bs, budget in item.token_budgets.items()
            ):
                raise ValueError(
                    f"SPS profile widths/budgets exceed draft block size {block}"
                )
        return default, mode_policies

    @classmethod
    def from_dict(cls, policy):
        names = ("verify_widths", "token_budgets")
        if set(policy) - set(names):
            raise ValueError("Unknown fields in DSpine SPS policy")
        return cls(**{
            name: {int(bs): value for bs, value in policy.get(name, {}).items()}
            for name in names
        })


def hardware_identity(device):
    import torch

    props = torch.cuda.get_device_properties(device)
    return {
        "name": props.name,
        "capability": [props.major, props.minor],
        "memory_bytes": props.total_memory,
    }


def execution_identity(
    tp_size,
    draft_tp_size,
    attention_backend=None,
    draft_attention_backend=None,
    block_size=None,
):
    identity = {
        "tp_size": int(tp_size),
        "draft_tp_size": int(draft_tp_size),
        "attention_backend": attention_backend,
        "draft_attention_backend": draft_attention_backend,
    }
    if block_size is not None and int(block_size) != 16:
        identity["block_size"] = int(block_size)
    return identity
