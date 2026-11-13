"""Post-hoc maps from draft confidence to conditional acceptance probability.

Sampled confidence comes from the temperature-scaled candidate distribution
along the actual predecessor chain, before top-k/top-p/min-p filtering.
Depth zero excludes the known anchor.
"""

import json
import math
from pathlib import Path

import torch


FORMAT = "dspine-confidence-calibration-v1"
SAMPLED_PROBABILITY_SOURCE = "pre_token_max_probability_actual_sampled_chain"


def load_calibration_tables(artifact, gamma=15):
    """Validate both serving families and materialize CPU lookup tensors once."""
    return {mode: model.probabilities for mode, model in load_calibration_models(artifact, gamma).items()}


def load_calibration_models(artifact, gamma=15):
    """Return lookup modules, including optional nonuniform bin boundaries."""
    if isinstance(artifact, (str, Path)):
        artifact = json.loads(Path(artifact).read_text())
    if "confidence_sources" in artifact:
        sources = artifact["confidence_sources"]
        if not isinstance(sources, dict) or sources.get("sampled") != SAMPLED_PROBABILITY_SOURCE:
            raise ValueError(
                "sampled calibration feature differs from the runtime pre-token "
                "maximum probability before truncation; recollect and refit "
                "calibration with pre_token_max_probability_actual_sampled_chain"
            )
    bins = artifact.get("lookup_bins")
    if artifact.get("gamma") != gamma or type(bins) is not int or bins < 2:
        raise ValueError("calibration gamma or lookup_bins differs from the serving block")
    if artifact.get("lookup_grid", "linear") not in ("linear", "logit"):
        raise ValueError("lookup_grid must be linear or logit")
    models = {}
    for mode in ("greedy", "sampled"):
        calibrator = ConfidenceCalibrator.from_artifact(artifact, mode, bins)
        if calibrator.kind != "table" or calibrator.probabilities.shape != (gamma, bins):
            raise ValueError("calibration needs a [gamma, lookup_bins] table for each mode")
        if artifact.get("lookup_grid") == "logit" and calibrator.boundaries.numel() != bins - 1:
            raise ValueError("logit lookup needs its nonuniform boundaries")
        models[mode] = calibrator
    return models


class ConfidenceCalibrator(torch.nn.Module):
    """Apply a fitted per-depth map to ``raw[..., gamma]`` on CPU or CUDA.

    Load/construct before graph capture, then move with ``.to(device)``. Forward
    contains no CPU transfers. Positive logit scales preserve ordering within
    each depth; isotonic lookup maps preserve weak ordering. The old empirical
    table is supported as an explicitly non-monotonic comparison baseline.
    """

    def __init__(self, model):
        super().__init__()
        self.kind = model["kind"]
        self.gamma = int(model["gamma"])
        if self.gamma < 1:
            raise ValueError("gamma must be positive")
        self.eps = float(model.get("eps", 1e-6))
        if not 0 < self.eps < 0.5:
            raise ValueError("eps must lie in (0, 0.5)")
        if self.kind == "temperature":
            temperatures = model["temperature"]
            if any(not math.isfinite(t) or t <= 0 for t in temperatures):
                raise ValueError("temperatures must be finite and positive")
            scale = [1 / t for t in temperatures]
            offset = [0.0] * self.gamma
        elif self.kind == "logit":
            scale, offset = model["scale"], model["offset"]
        elif self.kind in ("identity", "table"):
            scale, offset = [1.0] * self.gamma, [0.0] * self.gamma
        else:
            raise ValueError(f"unknown confidence map: {self.kind}")
        if len(scale) != self.gamma or len(offset) != self.gamma:
            raise ValueError("logit parameters must have one value per depth")
        if any(not math.isfinite(x) or x <= 0 for x in scale):
            raise ValueError("logit scales must be finite and positive")
        if any(not math.isfinite(x) for x in offset):
            raise ValueError("logit offsets must be finite")
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))
        self.register_buffer("offset", torch.tensor(offset, dtype=torch.float32))
        self.register_buffer("depth", torch.arange(self.gamma))
        if self.kind == "table":
            table = torch.tensor(model["probabilities"], dtype=torch.float32)
            if table.ndim != 2 or table.shape[0] != self.gamma or table.shape[1] < 2:
                raise ValueError("table must have shape [gamma, bins>=2]")
            if not bool(torch.isfinite(table).all()) or bool(((table < 0) | (table > 1)).any()):
                raise ValueError("table probabilities must lie in [0, 1]")
            if model.get("monotonic", False) and bool((table[:, 1:] < table[:, :-1]).any()):
                raise ValueError("declared monotonic table decreases")
        else:
            table = torch.empty((self.gamma, 0), dtype=torch.float32)
        self.register_buffer("probabilities", table)
        # Float32 merges distinct logit-grid boundaries near 1-EPS. These few
        # KiB stay double so CPU and CUDA bucketize share the exact partition.
        boundaries = torch.tensor(model.get("boundaries", []), dtype=torch.float64)
        if boundaries.numel():
            if self.kind != "table" or boundaries.shape != (table.shape[1] - 1,):
                raise ValueError("boundaries must have bins-1 entries for a table")
            if not bool(torch.isfinite(boundaries).all()) or bool((boundaries[1:] <= boundaries[:-1]).any()) or not (0 < float(boundaries[0]) and float(boundaries[-1]) < 1):
                raise ValueError("lookup boundaries must strictly increase within (0, 1)")
        self.register_buffer("boundaries", boundaries)

    @classmethod
    def from_artifact(cls, artifact, mode="greedy", lookup_bins=None):
        if isinstance(artifact, (str, Path)):
            artifact = json.loads(Path(artifact).read_text())
        if artifact.get("format") != FORMAT:
            raise ValueError("not a DSpine confidence calibration artifact")
        family = artifact["families"][mode]
        grid = artifact.get("lookup_grid", "linear")
        key = str(lookup_bins) if grid == "linear" else f"{grid}{lookup_bins}"
        model = family["model"] if lookup_bins is None else family["lookup_tables"][key]
        return cls(model)

    def forward(self, raw):
        if raw.shape[-1] != self.gamma:
            raise ValueError("confidence depth does not match the calibration")
        raw = raw.float().clamp(0.0, 1.0)
        if self.kind == "identity":
            return raw
        if self.kind == "table":
            if self.boundaries.numel():
                bins = torch.bucketize(raw.contiguous(), self.boundaries)
            else:
                bins = (raw * self.probabilities.shape[1]).long().clamp(0, self.probabilities.shape[1] - 1)
            return self.probabilities[self.depth, bins]
        # Clip in log space: forming float32(1-eps) first distorts the upper
        # endpoint enough to matter for a high STS temperature.
        upper = math.log1p(-self.eps)
        log_p = torch.log(raw.clamp_min(self.eps)).clamp_max(upper)
        log_not_p = torch.log((1 - raw).clamp_min(self.eps)).clamp_max(upper)
        return torch.sigmoid((log_p - log_not_p) * self.scale + self.offset)

    def prefix_survival(self, raw):
        """P(accepted >= k), k=1..gamma, with the anchor excluded."""
        return self(raw).cumprod(dim=-1)
