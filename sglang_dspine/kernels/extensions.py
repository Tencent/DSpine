"""Build the three sampling kernels from packaged source using PyTorch's cache."""

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def sampling_kernels():
    import torch
    from flashinfer.jit.env import FLASHINFER_INCLUDE_DIR
    from torch.utils.cpp_extension import load

    if torch.cuda.get_device_capability() != (9, 0):
        raise ValueError("Qwen3.8-27B sampling kernels require a Hopper GPU")
    root = Path(__file__).parent / "csrc"

    def build(name, folder, sources, flags):
        directory = root / folder
        return load(
            name=name,
            sources=[str(directory / name) for name in sources],
            extra_include_paths=[str(directory), str(FLASHINFER_INCLUDE_DIR)],
            extra_cflags=["-O3", "-std=c++20"],
            extra_cuda_cflags=[
                "-O3",
                "-std=c++20",
                "-gencode=arch=compute_90,code=sm_90"
                if folder == "mixed_sum"
                else "-gencode=arch=compute_90a,code=sm_90a",
                *flags,
            ],
        )

    air = build(
        "dspine_sparse_support",
        "sparse_support",
        ["bindings.cpp", "air_cut.cu"],
        ["--use_fast_math"],
    )
    topk = build(
        "dspine_conditional_topk",
        "conditional_topk",
        ["k_bindings.cpp", "conditional_topk.cu"],
        ["--use_fast_math"],
    )
    # The denominator must retain the original FP32 reduction order and rounding.
    reduction = build(
        "dspine_exact_sum",
        "mixed_sum",
        ["bindings.cpp", "mixed_exact_sum.cu"],
        ["--fmad=false", "--ftz=false"],
    )
    return air, topk, reduction
