# Copyright 2026 Tencent
# Licensed under the Apache License, Version 2.0.

"""Vocabulary-sharded head access for DSpine's unchanged refinement tail."""

import torch


def merge_topk(values, ids, k):
    """Global top-k with lower token IDs winning boundary ties."""
    order = torch.argsort(ids, dim=-1, stable=True)
    values, ids = values.gather(-1, order), ids.gather(-1, order)
    order = torch.argsort(values, dim=-1, descending=True, stable=True)[..., :k]
    return values.gather(-1, order), ids.gather(-1, order)


class TPHead:
    """Keep head rows sharded; exchange candidates and selected rows only.

    ``weight`` is the target's padded BF16 local head, with original-vocabulary
    rows first. All ranks use the same request rows and call collectives in the
    same order. The group follows the head, rather than a global default group.
    """

    def __init__(self, weight, start, end, global_vocab_size, tp_group):
        self.weight = weight
        self.start, self.end = int(start), int(end)
        self.global_vocab_size = int(global_vocab_size)
        self.group = tp_group
        self.world_size = int(tp_group.world_size)
        if (
            weight.ndim != 2
            or weight.dtype != torch.bfloat16
            or weight.stride(1) != 1
            or not 0 <= self.start <= self.end <= self.global_vocab_size
            or self.end - self.start > weight.shape[0]
            or weight.shape[0] == 0
        ):
            raise ValueError("Invalid DSpine vocabulary head shard")
        geometry = torch.tensor(
            [[self.start, self.end, weight.shape[0]]],
            device=weight.device,
            dtype=torch.int64,
        )
        geometry = self.group.all_gather(geometry, dim=0).cpu().tolist()
        cursor = 0
        for begin, finish, allocated in geometry:
            if begin != cursor or finish < begin or finish - begin > allocated:
                raise ValueError(
                    "DSpine head shards must partition the original vocabulary"
                )
            cursor = finish
        if cursor != self.global_vocab_size:
            raise ValueError("DSpine head shards do not cover the original vocabulary")
        self.geometry = tuple(tuple(row) for row in geometry)
        self.max_rows = max(row[2] for row in geometry)

    @property
    def num_local(self):
        return self.end - self.start

    def gather_vocab_rows(self, local):
        """Reconstruct a low-rank vocabulary table, excluding per-shard padding."""
        if local.shape[0] != self.num_local:
            raise ValueError("DSpine local vocabulary table has the wrong row count")
        padded = local.new_zeros((self.max_rows, *local.shape[1:]))
        padded[: self.num_local].copy_(local)
        gathered = self.group.all_gather(padded, dim=0).view(
            self.world_size, self.max_rows, *local.shape[1:]
        )
        result = local.new_empty((self.global_vocab_size, *local.shape[1:]))
        for rank, (begin, end, _) in enumerate(self.geometry):
            result[begin:end].copy_(gathered[rank, : end - begin])
        return result

    def global_topk(self, local_values, local_ids, k):
        """Merge equally padded local candidate blocks without gathering logits."""
        if local_values.shape != local_ids.shape:
            raise ValueError("DSpine local candidate values and IDs must match")
        values = self.group.all_gather(local_values, dim=-1)
        ids = self.group.all_gather(local_ids, dim=-1)
        return merge_topk(values, ids, k)

    def selected_rows(self, ids):
        """Replicate only the selected head rows, preserving their BF16 values."""
        owner = (ids >= self.start) & (ids < self.end)
        local_ids = (ids - self.start).clamp(0, self.weight.shape[0] - 1)
        rows = self.weight[local_ids].masked_fill(~owner[..., None], 0).contiguous()
        return self.group.all_reduce(rows)

    def rescore(self, ids, hidden):
        """Owner computes the original BF16 head dot; exchange scalar scores."""
        owner = (ids >= self.start) & (ids < self.end)
        local_ids = (ids - self.start).clamp(0, self.weight.shape[0] - 1).contiguous()
        if hidden.device.type == "cuda":
            from sglang_dspine.layers.fp8 import gather as kernels

            scores = kernels.rescore(torch, self.weight, local_ids, hidden)
        else:
            scores = (self.weight[local_ids].float() * hidden[:, None].float()).sum(-1)
            scores = scores.to(hidden.dtype)
        scores.masked_fill_(~owner, 0)
        return self.group.all_reduce(scores)
