"""Prepare factorized proposal logits after native KV pools are allocated."""

import torch

from sglang_dspine.sampling.factorized.graph import ProposalGraphs


def prepare(sampler):
    saved = {
        name: getattr(sampler, name).clone()
        for name in ("raw", "rms", "query", "anchors", "candidate_out")
    }
    rng = torch.cuda.get_rng_state(sampler.raw.device).clone()
    try:
        sampler.raw.fill_(1.0)
        sampler.rms.zero_()
        sampler.query.zero_()
        sampler.anchors.zero_()
        sampler.candidate_out.copy_(
            torch.arange(sampler.k, device=sampler.raw.device).expand_as(
                sampler.candidate_out
            )
        )
        cache = ProposalGraphs(sampler)
    finally:
        for name, value in saved.items():
            getattr(sampler, name).copy_(value)
    if not torch.equal(rng, torch.cuda.get_rng_state(sampler.raw.device)):
        raise RuntimeError(
            "Deterministic proposal graph preparation changed the RNG state"
        )
    return cache
