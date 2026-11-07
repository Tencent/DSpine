"""DSpine inference kernels and graph-owned buffers."""

import importlib
import threading
from contextlib import contextmanager

_PREPARE_LOCK = threading.RLock()


class PrivateFIWorkspace:
    def __init__(self, torch, stream, modules=None):
        self.torch = torch
        self.stream = stream
        self.thread = threading.get_ident()
        self.modules = modules
        self.buffers = {}
        self.retired = []
        self.frozen = False
        self.closed = False
        self.bound = False
        self.originals = {}
        self.helpers_restored = False

    def workspace(self, name, num_bytes, device, zero_init=False):
        assert not self.closed
        key = (name, str(device))
        buf = self.buffers.get(key)
        if buf is None or buf.numel() < num_bytes:
            if self.frozen:
                raise RuntimeError(
                    "Unexpected FI scratch growth during capture: " + name
                )
            if buf is not None:
                self.retired.append(buf)
            buf = (self.torch.zeros if zero_init else self.torch.empty)(
                num_bytes, device=device, dtype=self.torch.uint8
            )
            self.buffers[key] = buf
        return buf

    def freeze(self):
        self.frozen = True

    @contextmanager
    def bind(self):
        assert not self.closed and (not self.bound)
        with _PREPARE_LOCK:
            modules = self.modules or [
                importlib.import_module("flashinfer.topk"),
                importlib.import_module("flashinfer.sampling"),
            ]
            self.originals = {m: m._get_cache_buf for m in modules}
            self.bound = True

            def wrap(original):

                def get(name, num_bytes, device, zero_init=False):
                    if threading.get_ident() == self.thread and int(
                        self.torch.cuda.current_stream(device).cuda_stream
                    ) == int(self.stream.cuda_stream):
                        return self.workspace(name, num_bytes, device, zero_init)
                    return original(name, num_bytes, device, zero_init=zero_init)

                return get

            try:
                for m, original in self.originals.items():
                    m._get_cache_buf = wrap(original)
                yield self
            finally:
                for m, original in self.originals.items():
                    m._get_cache_buf = original
                self.bound = False
                self.helpers_restored = all(
                    (m._get_cache_buf is f for (m, f) in self.originals.items())
                )

    def stats(self):
        return dict(
            private_bytes=sum(
                (b.numel() * b.element_size() for b in self.buffers.values())
            ),
            private_pointers={
                str(k): int(b.data_ptr()) for (k, b) in self.buffers.items()
            },
            retired_pointer_count=len(self.retired),
            capture_stream=int(self.stream.cuda_stream),
            owner_thread=self.thread,
            strong_refs_retained_until_graph_reset=True,
            FI_helpers_restored=self.helpers_restored,
        )

    def release_after_reset(self):
        assert not self.bound, "Do not release scratch during warm/capture"
        self.buffers.clear()
        self.retired.clear()
        self.closed = True
