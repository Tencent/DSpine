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
    return result


def runner(name, backend=None, **overrides):
    fields = dict(
        draft_attention_backend=name,
        attn_backend=backend,
        sliding_window_size=8191,
        is_hybrid_swa=False,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize("name", ["fa3", "triton"])
def test_supported_backend_preserves_swa_contract(name):
    validate_native_swa_runner(runner(name), served_block_size=16)
    with pytest.raises(ValueError, match="window_left"):
        validate_native_swa_runner(
            runner(name, sliding_window_size=-1), served_block_size=16
        )
    with pytest.raises(ValueError, match="dense draft KV"):
        validate_native_swa_runner(
            runner(name, is_hybrid_swa=True), served_block_size=16
        )
    with pytest.raises(ValueError, match="block size16"):
        validate_native_swa_runner(runner(name), served_block_size=8)


@pytest.mark.parametrize("name", [None, "fa4", "flashinfer"])
def test_other_draft_backends_remain_rejected(name):
    with pytest.raises(ValueError, match="fa3 or triton"):
        validate_native_swa_runner(runner(name), served_block_size=16)


@pytest.mark.parametrize("name", ["fa3", "triton"])
def test_selection_checks_the_resolved_backend(name, backend_types):
    backend = backend_types[name]()
    if name == "fa3":
        backend.fa_impl_ver = 3
    validate_native_swa_backend(runner(name, backend))
    with pytest.raises(ValueError, match="incompatible"):
        validate_native_swa_backend(runner(name, object()))


def test_fa3_cannot_resolve_to_fa4(backend_types):
    backend = backend_types["fa3"]()
    backend.fa_impl_ver = 4
    with pytest.raises(ValueError, match="incompatible"):
        validate_native_swa_backend(runner("fa3", backend))


@pytest.mark.parametrize("name", ["fa3", "triton"])
def test_backend_cannot_resolve_to_the_other_implementation(name, backend_types):
    backend = backend_types["triton" if name == "fa3" else "fa3"]()
    with pytest.raises(ValueError, match="incompatible"):
        validate_native_swa_backend(runner(name, backend))
