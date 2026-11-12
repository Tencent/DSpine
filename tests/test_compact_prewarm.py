"""The startup compact specialization preserves sampler state and RNG."""

from types import SimpleNamespace

import pytest
import torch

from sglang_dspine.runtime.worker import DSpineWorkerV2


@pytest.mark.parametrize("fail", [False, True])
def test_compact_rehearsal_restores_rng_and_proposal_state(monkeypatch, fail):
    bs, width, k = 3, 16, 16
    worker = object.__new__(DSpineWorkerV2)
    sampler = SimpleNamespace(k=k, _round_bs=9, _round_sampled=False,
        candidate_out=torch.full((bs, width - 1, k), 67, dtype=torch.int64),
        q_out=torch.full((bs, width - 1, k), .17))
    worker._dspine_sampler = sampler
    worker._dspine_verify = lambda **kwargs: None
    seen = {}
    def compact(**kwargs):
        seen.update(kwargs)
        torch.rand(17)
        if fail:
            raise RuntimeError("synthetic prewarm failure")
    worker.target_distribution = SimpleNamespace(sample_prefix=compact)
    for name in ("synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda: {
        "allocated_bytes.all.current": 100, "allocated_bytes.all.peak": 700})
    torch.manual_seed(731)
    state = torch.random.get_rng_state().clone()
    def run():
        return worker._measure_sampling_peak(candidates=torch.zeros(bs, width, dtype=torch.int64),
            target_logits=torch.zeros(bs * width, 31), sampling_info=object())
    if fail:
        with pytest.raises(RuntimeError, match="synthetic"):
            run()
    else:
        assert run() == 600
    assert torch.equal(state, torch.random.get_rng_state())
    assert sampler._round_bs == 9 and not sampler._round_sampled
    assert bool((sampler.candidate_out == 67).all())
    assert bool((sampler.q_out == .17).all())
    assert torch.equal(seen["row_offsets"], torch.tensor([0, 16, 32], dtype=torch.int32))
    assert seen["prefix_lens"].dtype == torch.int64
    assert sampler.candidate_out.dtype == torch.int64
