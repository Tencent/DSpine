# Copyright 2026 Tencent
# Licensed under the Apache License, Version 2.0.

"""Mandatory agreement within the TP ranks serving one DSpine batch."""

import secrets

import torch


class DSpineTpCoordinator:
    """The worker supplies its batch's TP group before entering a draft scope."""

    def __init__(self, group):
        self.group = group
        self.enabled = group.world_size > 1
        self.is_leader = group.rank_in_group == 0
        self._accept_packets = {}
        self._layout_packets = {}

    def sync_tensor(self, value):
        if self.enabled and value.numel():
            self.group.broadcast(value, src=0)
        return value

    def request_seeds(self, reqs):
        """Share newly admitted unseeded requests once, then reuse their seeds."""
        seeds = [
            req.sampling_params.sampling_seed
            if req.sampling_params.sampling_seed is not None
            else getattr(req, "_dspine_random_seed", None)
            for req in reqs
        ]
        if any(seed is None for seed in seeds):
            seeds = (
                [secrets.randbits(63) if seed is None else seed for seed in seeds]
                if self.is_leader
                else None
            )
            if self.enabled:
                seeds = self.group.broadcast_object(seeds, src=0)
            for req, seed in zip(reqs, seeds):
                if req.sampling_params.sampling_seed is None:
                    req._dspine_random_seed = seed
        return seeds

    def uniforms(self, shape, device):
        """Fallback/probe draws also use the leader's coins, outside CUDA graphs."""
        value = (
            torch.rand(shape, device=device, dtype=torch.float32)
            if self.is_leader or not self.enabled
            else torch.empty(shape, device=device, dtype=torch.float32)
        )
        return self.sync_tensor(value)

    def sync_proposal(self, sampler, bs):
        """Keep the sampled chain and its actual conditional Q together."""
        if not self.enabled or not bs:
            return
        self.sync_tensor(sampler.out[: bs * (sampler.block_size - 1)])
        self.sync_tensor(sampler.candidate_out[:bs])
        if sampler.round_sampled:
            self.sync_tensor(sampler.q_out[:bs])
        if sampler.confidence_out is not None:
            self.sync_tensor(sampler.confidence_out[:bs])

    def sync_accept(self, *values):
        """Share existing acceptance buffers in place, before cache updates."""
        self._sync_values(
            tuple(value for value in values if value is not None), self._accept_packets
        )
        return values

    def sync_layout(self, layout, index):
        """Share a replayed planner's outputs without replacing static buffers."""
        if not self.enabled:
            return
        values = (
            layout.verify_lens,
            layout.extend_start_loc,
            layout.qo_indptr_device,
            index,
        )
        self._sync_values(values, self._layout_packets)

    def _sync_values(self, values, packets):
        """One integer packet; copies retain graph/double-buffer storage and dtypes."""
        if not self.enabled or not values:
            return
        size = sum(value.numel() for value in values)
        key = (values[0].device, size)
        packet = packets.get(key)
        if packet is None:
            packet = torch.empty(size, device=values[0].device, dtype=torch.int64)
            packets[key] = packet
        offset = 0
        for value in values:
            end = offset + value.numel()
            packet[offset:end].copy_(value.reshape(-1))
            offset = end
        self.sync_tensor(packet)
        offset = 0
        for value in values:
            end = offset + value.numel()
            value.copy_(packet[offset:end].view_as(value))
            offset = end
