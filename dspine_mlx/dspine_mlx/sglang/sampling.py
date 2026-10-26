"""Per-request sampling parameters and request admission checks."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

DEFAULT_SAMPLING_SEED = 42


@dataclass(frozen=True)
class RequestSampling:
    greedy: bool
    temperature: float
    inv_temp: float
    top_k: int
    top_p: float
    min_p: float
    chain_top_k: int
    target_top_k: int
    seed: int | None

    @classmethod
    def from_req(
        cls,
        req,
        *,
        head_topk: int,
        vocab_size: int,
        deterministic_seeding: bool = False,
    ) -> "RequestSampling":
        sp = req.sampling_params
        top_k = int(sp.top_k)
        temperature = float(sp.temperature)
        # SGLang normalizes temperature ~0 to top_k=1; accept either spelling.
        greedy = top_k == 1 or temperature <= 1e-5
        unrestricted = top_k >= int(vocab_size)
        chain_top_k = (
            int(head_topk)
            if unrestricted
            else min(max(top_k, 1), int(head_topk))
        )
        target_top_k = int(vocab_size) if unrestricted else max(top_k, 1)
        seed = None
        if deterministic_seeding:
            seed = (
                int(sp.sampling_seed)
                if sp.sampling_seed is not None
                else DEFAULT_SAMPLING_SEED
            )
        return cls(
            greedy=greedy,
            temperature=temperature,
            inv_temp=1.0 / max(temperature, 1e-5),
            top_k=top_k,
            top_p=float(sp.top_p),
            min_p=float(sp.min_p),
            chain_top_k=chain_top_k,
            target_top_k=target_top_k,
            seed=seed,
        )


def seed_key(seed: int, round_index: int) -> mx.array:
    """Deterministic per-request, per-round RNG key."""
    mixed = (int(seed) * 0x9E3779B1 + int(round_index) * 0x85EBCA77) & 0xFFFFFFFF
    return mx.random.key(mixed)


def validate_request(req, *, head_topk: int) -> str | None:
    """Return an error message for request features the MLX plugin lacks."""
    sp = req.sampling_params
    if getattr(req, "grammar", None) is not None:
        return "DSPINE MLX does not support grammar-constrained decoding."
    if getattr(sp, "logit_bias", None):
        return "DSPINE MLX does not support logit_bias."
    if (
        float(getattr(sp, "frequency_penalty", 0.0)) != 0.0
        or float(getattr(sp, "presence_penalty", 0.0)) != 0.0
        or float(getattr(sp, "repetition_penalty", 1.0)) != 1.0
    ):
        return "DSPINE MLX does not support frequency/presence/repetition penalties."
    if getattr(req, "return_logprob", False):
        return "DSPINE MLX does not support return_logprob yet."
    if getattr(req, "return_hidden_states", False):
        return "DSPINE MLX does not support return_hidden_states."
    if getattr(req, "return_sampling_mask", False):
        return "DSPINE MLX does not support return_sampling_mask."
    if getattr(req, "custom_logit_processor", None) is not None:
        return "DSPINE MLX does not support custom logit processors."
    if getattr(req, "input_embeds", None) is not None:
        return "DSPINE MLX does not support input embeddings."
    if getattr(req, "multimodal_inputs", None) is not None:
        return "DSPINE MLX does not support multimodal inputs."
    if int(sp.top_k) < 1:
        return f"Invalid top_k={sp.top_k}."
    if int(head_topk) < 2:
        return f"Invalid draft head_topk={head_topk}."
    return None
