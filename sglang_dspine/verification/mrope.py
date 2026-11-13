"""MRoPE positions for packed speculative prefixes."""

import torch


def ragged_mrope_positions(positions, qo_indptr, deltas=None):
    """Apply each request's native MRoPE delta to its packed token rows."""
    positions = positions.to(torch.int64).reshape(-1)
    if deltas is not None:
        deltas = deltas.reshape(-1).to(device=positions.device, dtype=torch.int64)
        token = torch.arange(
            positions.numel(), device=positions.device, dtype=torch.int32
        )
        request = torch.searchsorted(qo_indptr[1:].contiguous(), token, right=True)
        valid = request < deltas.numel()
        delta = deltas.index_select(0, request.clamp(max=deltas.numel() - 1))
        positions = torch.where(valid, positions + delta, torch.zeros_like(positions))
    return positions.unsqueeze(0).repeat(3, 1)


def around_spec_mrope_positions(
    original, forward_batch, model_runner, batch, seq_positions=None
):
    from sglang_dspine.verification.ragged import ragged_active

    spec_info = getattr(batch, "spec_info", None)
    layout = getattr(spec_info, "ragged_verify_layout", None)
    if not ragged_active() or layout is None:
        return original(forward_batch, model_runner, batch, seq_positions)
    from sglang_dspine.spec import is_dspine

    if not is_dspine(model_runner.spec_algorithm):
        return original(forward_batch, model_runner, batch, seq_positions)
    if seq_positions is None:
        seq_positions = spec_info.positions
    mm_inputs = batch.multimodal_inputs
    deltas = None
    if any(mm_input is not None for mm_input in mm_inputs):
        # Match ForwardBatch.compute_spec_mrope_positions: text rows have a
        # zero delta; multimodal rows use mrope_position_delta.squeeze(0).
        deltas = torch.stack(
            [
                torch.zeros(1, dtype=torch.int64)
                if mm_input is None
                else mm_input.mrope_position_delta.squeeze(0)
                for mm_input in mm_inputs
            ],
            dim=0,
        ).to(device=model_runner.device)
    forward_batch.mrope_positions = ragged_mrope_positions(
        seq_positions, layout.qo_indptr_device, deltas
    )
