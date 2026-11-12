"""Use native launch geometry for the empty-history ReplaySSM specialization."""

import inspect
from functools import lru_cache

LAUNCH_KEYS = ("block_v", "num_warps", "num_stages", "nk")


def _tensor(x, shape, dtype, device):
    return (
        x is not None
        and tuple(x.shape) == tuple(shape)
        and str(x.dtype) == dtype
        and x.device == device
    )


def select_launch(algorithm, kwargs, defaults):
    """Specialize the history work, inheriting SGLang's launch parameters."""
    name = algorithm if isinstance(algorithm, str) else getattr(algorithm, "name", None)
    if str(name).upper() != "DSPINE" or kwargs.get("launch_mode") != "verify":
        return None
    q, v, indptr = (kwargs.get(key) for key in ("q", "v", "query_start_loc"))
    if any(x is None for x in (q, v, indptr)):
        return None
    if q.ndim != 3 or v.ndim != 3 or indptr.ndim != 1 or q.device.type != "cuda":
        return None
    n, heads, key_dim = q.shape
    _, value_heads, value_dim = v.shape
    bs = indptr.numel() - 1
    spec, ring = kwargs.get("max_spec_len"), kwargs.get("max_cache_len")
    if (
        bs < 1 or heads < 1 or value_heads % heads
        or not isinstance(spec, int) or not isinstance(ring, int)
        or not 1 <= spec <= ring or ring & (ring - 1)
        or not 1 <= n <= bs * spec
    ):
        return None
    device = q.device
    shapes = {
        "q": (n, heads, key_dim), "k": (n, heads, key_dim),
        "v": (n, value_heads, value_dim), "out": (n, value_heads, value_dim),
        "a": (n, value_heads), "b": (n, value_heads),
    }
    if not all(_tensor(kwargs.get(k), shape, "torch.bfloat16", device)
               for k, shape in shapes.items()):
        return None
    if not _tensor(kwargs.get("A_log"), (value_heads,), "torch.float32", device):
        return None
    bias = kwargs.get("dt_bias")
    if bias is None or str(bias.dtype) not in ("torch.bfloat16", "torch.float32"):
        return None
    if not _tensor(bias, (value_heads,), str(bias.dtype), device):
        return None
    checkpoint, d = kwargs.get("checkpoint_state"), kwargs.get("d_cache")
    if checkpoint is None or checkpoint.ndim != 4 or d is None or d.ndim != 4:
        return None
    slots = d.shape[0]
    if not _tensor(checkpoint, (checkpoint.shape[0], value_heads, value_dim, key_dim),
                   "torch.float32", device):
        return None
    if slots < bs:
        return None
    expected = {
        "d_cache": ((slots, value_heads, ring, value_dim), "torch.bfloat16"),
        "k_cache": ((slots, heads, ring, key_dim), "torch.bfloat16"),
        "g_cache": ((slots, value_heads, ring), "torch.float32"),
        "rawv_cache": ((slots, value_heads, ring, value_dim), "torch.bfloat16"),
        "rawk_cache": ((slots, heads, ring, key_dim), "torch.bfloat16"),
        "query_start_loc": ((bs + 1,), "torch.int32"),
        "replay_indices": ((bs,), "torch.int64"),
        "write_pos": ((slots,), "torch.int32"),
        "cache_base": ((slots,), "torch.int32"),
        "is_flush": ((slots,), "torch.int8"),
    }
    if not all(_tensor(kwargs.get(k), shape, dtype, device)
               for k, (shape, dtype) in expected.items()):
        return None
    indices = kwargs.get("ssm_state_indices")
    if indices is None or str(indices.dtype) not in ("torch.int32", "torch.int64"):
        return None
    if not _tensor(indices, (bs,), str(indices.dtype), device):
        return None
    if kwargs.get("beta_cache") is not None:
        return None
    effective = lambda key: kwargs.get(key, defaults.get(key))
    if effective("null_block_id") != -1:
        return None
    return tuple(effective(key) for key in LAUNCH_KEYS)


@lru_cache(maxsize=None)
def native_defaults(original):
    return {
        k: p.default
        for k, p in inspect.signature(original).parameters.items()
        if p.default is not inspect.Parameter.empty
    }
