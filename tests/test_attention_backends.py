"""Draft backend dispatch retains the trained causal sliding-window layout."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from sglang_dspine.layers.attention import (
    validate_native_swa_backend,
    validate_native_swa_runner,
)


@pytest.fixture
def backend_types(monkeypatch):
    """Keep configuration tests independent of CUDA backend module imports."""
    result = {}
    for name, module, class_name in (
        ("fa3", "flashattention_backend", "FlashAttentionBackend"),
        ("triton", "triton_backend", "TritonAttnBackend"),
    ):
        path = f"sglang.srt.layers.attention.{module}"
        stub = ModuleType(path)
        result[name] = type(class_name, (), {})
        setattr(stub, class_name, result[name])
        monkeypatch.setitem(sys.modules, path, stub)
    return {**result, "fa4": result["fa3"]}


def runner(name, backend=None, **overrides):
    fields = dict(
        draft_attention_backend=name,
        attn_backend=backend,
        sliding_window_size=8191,
        is_hybrid_swa=False,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize("name", ["fa3", "fa4", "triton"])
def test_supported_backend_preserves_swa_contract(name):
    for served in (2, 6, 8, 12, 16):
        validate_native_swa_runner(runner(name), served_block_size=served)
    with pytest.raises(ValueError, match="window_left"):
        validate_native_swa_runner(
            runner(name, sliding_window_size=-1), served_block_size=16
        )
    with pytest.raises(ValueError, match="dense draft KV"):
        validate_native_swa_runner(
            runner(name, is_hybrid_swa=True), served_block_size=16
        )
    for bad in (0, 1, 17, True, "8"):
        with pytest.raises(ValueError, match="block sizes 2..16"):
            validate_native_swa_runner(runner(name), served_block_size=bad)


@pytest.mark.parametrize("name", [None, "flashinfer"])
def test_other_draft_backends_remain_rejected(name):
    with pytest.raises(ValueError, match="fa3, fa4, or triton"):
        validate_native_swa_runner(runner(name), served_block_size=16)


@pytest.mark.parametrize("name, version", [("fa3", 3), ("fa4", 4), ("triton", None)])
def test_selection_checks_the_resolved_backend(name, version, backend_types):
    backend = backend_types[name]()
    if version is not None:
        backend.fa_impl_ver = version
    validate_native_swa_backend(runner(name, backend))
    with pytest.raises(ValueError, match="incompatible"):
        validate_native_swa_backend(runner(name, object()))


@pytest.mark.parametrize("name, wrong_version", [("fa3", 4), ("fa4", 3)])
def test_flash_attention_cannot_silently_change_version(
    name, wrong_version, backend_types
):
    backend = backend_types[name]()
    backend.fa_impl_ver = wrong_version
    with pytest.raises(ValueError, match="incompatible"):
        validate_native_swa_backend(runner(name, backend))


@pytest.mark.parametrize(
    "name, other", [("fa3", "triton"), ("fa4", "triton"), ("triton", "fa4")]
)
def test_backend_cannot_resolve_to_the_other_implementation(name, other, backend_types):
    backend = backend_types[other]()
    with pytest.raises(ValueError, match="incompatible"):
        validate_native_swa_backend(runner(name, backend))
