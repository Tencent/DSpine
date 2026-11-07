"""Commit the verified prefix and exact prefix-cache SSM/convolution boundary."""

from __future__ import annotations


def commit_gdn_replay(
    worker, *, batch, commit_lens, seq_lens_pre_verify, seq_lens_post_verify
) -> bool:
    """Commit consumed root + accepted drafts; the next bonus is not consumed."""
    runner = worker.target_worker.model_runner
    req_pool = runner.req_to_token_pool
    pool = getattr(req_pool, "mamba_pool", None)
    if pool is None or not getattr(pool, "enable_linear_replayssm_spec", False):
        return False
    if getattr(pool, "replayssm_is_kda", False) or getattr(
        pool, "replayssm_spec_fold", False
    ):
        raise RuntimeError(
            "DSPine 0.5.22 replay requires the GDN compact circular pool"
        )
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

    if type(req_pool) is not HybridReqToTokenPool:
        raise RuntimeError("DSPine replay requires the standard HybridReqToTokenPool")
    if batch.forward_mode.is_idle() or commit_lens.numel() == 0:
        return True
    import torch
    from sglang.srt.speculative.spec_utils import commit_mamba_states_after_verify

    state = req_pool.get_speculative_mamba2_params_all_layers()
    if state.temporal.dtype != torch.float32 or state.intermediate_ssm is not None:
        raise RuntimeError(
            "DSPine replay requires fp32 checkpoints without snapshot SSM state"
        )
    for name in ("replayssm_d", "replayssm_k", "replayssm_g"):
        if getattr(state, name, None) is None:
            raise RuntimeError(f"DSPine replay missing native history {name}")
    for name in (
        "replayssm_spec_write_pos",
        "replayssm_cache_base",
        "replayssm_is_flush",
    ):
        if getattr(pool, name, None) is None:
            raise RuntimeError(f"DSPine replay missing native cursor {name}")
    (bs, block) = (int(commit_lens.shape[0]), int(worker.block_size))
    indices = getattr(worker, "_dspine_replay_chain_indices", None)
    if indices is None or indices.shape[0] < bs:
        indices = torch.arange(
            bs * block, device=commit_lens.device, dtype=torch.int64
        ).view(bs, block)
        worker._dspine_replay_chain_indices = indices
    last_correct_step_indices = commit_lens.to(torch.int64) - 1
    mamba_steps_to_track = None
    if batch.mamba_track_indices is not None:
        from sglang.srt.runtime_context import mamba_track_grid

        interval = mamba_track_grid(batch.tree_cache.page_size)
        crosses = seq_lens_pre_verify // interval != seq_lens_post_verify // interval
        point = seq_lens_post_verify // interval * interval
        step = torch.clamp(point - seq_lens_pre_verify - 1, min=0)
        valid = crosses & (step < commit_lens.to(step.dtype))
        mamba_steps_to_track = torch.where(
            valid, step.to(torch.int64), torch.full_like(step, -1, dtype=torch.int64)
        )
    commit_mamba_states_after_verify(
        worker.target_worker,
        batch,
        accept_lens=commit_lens,
        accept_index=indices[:bs],
        draft_token_num=block,
        prepared_step_indices=(last_correct_step_indices, mamba_steps_to_track),
    )
    return True
