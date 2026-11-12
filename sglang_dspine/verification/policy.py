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
    def load(cls, path, hardware):
        return cls.load_with_modes(path, hardware)[0]

    @classmethod
    def load_with_modes(cls, path, hardware):
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
            return cls(dict(enumerate(widths, 1)), {8: budget}), {}
        modes = data.get("mode_policies", {})
        if not isinstance(modes, dict) or set(modes) - {"greedy", "sampled"}:
            raise ValueError("SPS mode_policies only supports greedy and sampled")
        return cls.from_dict(policy), {mode: cls.from_dict(value) for mode, value in modes.items()}

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
