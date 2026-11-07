"""Fixed GDN launch geometry for the supported batch sizes."""

import inspect
from functools import lru_cache

LAUNCH_KEYS = ("block_v", "num_warps", "num_stages", "nk")
DSPINE = {
    1: (32, 2, 2, 2),
    **{b: (64, 2, 2, 4) for b in range(2, 6)},
    **{b: (128, 2, 2, 4) for b in range(6, 9)},
}
TABLES = {"DSPINE": DSPINE}


def algorithm_name(algorithm):
    name = algorithm if isinstance(algorithm, str) else getattr(algorithm, "name", None)
    return name.upper() if isinstance(name, str) else None


def _tensor(x, shape, dtype, device):
    """Only Python tensor metadata; no item/cpu/tolist/copy/synchronization."""
    return (
        x is not None
        and tuple(x.shape) == tuple(shape)
        and (str(x.dtype) == dtype)
        and (x.device == device)
    )


def select_launch(algorithm, kwargs, native_defaults):
    """Return a tuple only for the source-backed static production metadata."""
    mode = algorithm_name(algorithm)
    if mode not in TABLES or kwargs.get("launch_mode") != "verify":
        return None
    qsl = kwargs.get("query_start_loc")
    if qsl is None or len(qsl.shape) != 1:
        return None
    bs = qsl.shape[0] - 1
    if bs not in TABLES[mode]:
        return None
    spec = kwargs.get("max_spec_len")
    if kwargs.get("max_cache_len") != 16 or spec not in (
        (12, 16) if mode == "DSPINE" else (8,)
    ):
        return None
    q = kwargs.get("q")
    if q is None or len(q.shape) != 3 or q.device.type != "cuda":
        return None
    (n, device) = (q.shape[0], q.device)
    if not 1 <= n <= bs * spec:
        return None
    if not all(
        (
            _tensor(kwargs.get(key), shape, dtype, device)
            for (key, shape, dtype) in (
                ("q", (n, 16, 128), "torch.bfloat16"),
                ("k", (n, 16, 128), "torch.bfloat16"),
                ("v", (n, 48, 128), "torch.bfloat16"),
                ("a", (n, 48), "torch.bfloat16"),
                ("b", (n, 48), "torch.bfloat16"),
                ("out", (n, 48, 128), "torch.bfloat16"),
                ("A_log", (48,), "torch.float32"),
            )
        )
    ):
        return None
    bias = kwargs.get("dt_bias")
    if bias is None or str(bias.dtype) not in ("torch.bfloat16", "torch.float32"):
        return None
    if not _tensor(bias, (48,), str(bias.dtype), device):
        return None
    (checkpoint, d) = (kwargs.get("checkpoint_state"), kwargs.get("d_cache"))
    if (
        checkpoint is None
        or len(checkpoint.shape) != 4
        or d is None
        or (len(d.shape) != 4)
    ):
        return None
    if not _tensor(
        checkpoint, (checkpoint.shape[0], 48, 128, 128), "torch.float32", device
    ):
        return None
    slots = d.shape[0]
    if slots < bs or not all(
        (
            _tensor(kwargs.get(key), shape, dtype, device)
            for (key, shape, dtype) in (
                ("d_cache", (slots, 48, 16, 128), "torch.bfloat16"),
                ("k_cache", (slots, 16, 16, 128), "torch.bfloat16"),
                ("g_cache", (slots, 48, 16), "torch.float32"),
                ("rawv_cache", (slots, 48, 16, 128), "torch.bfloat16"),
                ("rawk_cache", (slots, 16, 16, 128), "torch.bfloat16"),
                ("query_start_loc", (bs + 1,), "torch.int32"),
                ("replay_indices", (bs,), "torch.int64"),
                ("write_pos", (slots,), "torch.int32"),
                ("cache_base", (slots,), "torch.int32"),
                ("is_flush", (slots,), "torch.int8"),
            )
        )
    ):
        return None
    indices = kwargs.get("ssm_state_indices")
    if indices is None or str(indices.dtype) not in ("torch.int32", "torch.int64"):
        return None
    if not _tensor(indices, (bs,), str(indices.dtype), device):
        return None
    if kwargs.get("beta_cache") is not None:
        return None
    effective = lambda name: kwargs.get(name, native_defaults.get(name))
    if (
        effective("use_qk_l2norm_in_kernel") is not True
        or effective("bs_min") != 4
        or effective("null_block_id") != -1
        or (effective("dot_precision") != "tf32")
        or (effective("scale") not in (None, 128 ** (-0.5)))
    ):
        return None
    return TABLES[mode][bs]


@lru_cache(maxsize=None)
def native_defaults(original):
    return {
        k: p.default
        for (k, p) in inspect.signature(original).parameters.items()
        if p.default is not inspect.Parameter.empty
    }
