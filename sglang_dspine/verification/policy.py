"""Measured verify policy; draft width and rejection sampling stay unchanged."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SpsPolicy:
    # Unprofiled engines verify the full block, as in DSpark's flat-cost fallback.
    uniform_widths: tuple[int, ...] = (16,) * 7
    batch8_tokens: int = 128

    def __post_init__(self):
        if len(self.uniform_widths) != 7 or any(
            type(w) is not int or not 2 <= w <= 16 for w in self.uniform_widths
        ):
            raise ValueError("SPS requires seven integer widths in [2,16] for BS1–7")
        if type(self.batch8_tokens) is not int or self.batch8_tokens not in range(
            16, 129, 16
        ):
            raise ValueError("BS8 SPS budget must be a captured tier: 16,32,...,128")

    def to_dict(self):
        return {
            "uniform_widths": list(self.uniform_widths),
            "batch8_tokens": self.batch8_tokens,
        }

    @classmethod
    def load(cls, path, hardware):
        if not path:
            return cls()
        data = json.loads(Path(path).read_text())
        if data.get("format") != "dspine-sps-v1":
            raise ValueError("Expected a DSpine SPS profile from scripts/tune_sps.py")
        if data.get("hardware") != hardware:
            raise ValueError(
                "SPS profile hardware differs; rerun scripts/tune_sps.py on this GPU"
            )
        return cls(**data["policy"])


def hardware_identity(device):
    import torch

    props = torch.cuda.get_device_properties(device)
    return {
        "name": props.name,
        "capability": [props.major, props.minor],
        "memory_bytes": props.total_memory,
    }
