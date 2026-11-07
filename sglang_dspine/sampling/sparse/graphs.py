"""DSpine inference kernels and graph-owned buffers."""

import importlib
import threading
from contextlib import contextmanager
from types import SimpleNamespace

_PREPARE_LOCK = threading.RLock()
_ACTIVE_CAPTURE_STREAMS = set()


class _Entry:
    def __init__(self, cache, logits, key):
        self.cache = cache
        self.key = key
        (self.bs, self.device, self.caller_handle, self.thread, self.mode) = key
        torch = cache.torch
        self.caller_stream = torch.cuda.current_stream(logits.device)
        for _ in range(64):
            candidate = torch.cuda.Stream(device=logits.device)
            stream_key = (str(logits.device), int(candidate.cuda_stream))
            native_key = (
                self.bs * 16,
                str(logits.device),
                self.thread,
                int(candidate.cuda_stream),
            )
            if (
                stream_key not in _ACTIVE_CAPTURE_STREAMS
                and candidate != self.caller_stream
                and (native_key not in cache.producer._buffers)
            ):
                self.capture_stream = candidate
                self.stream_key = stream_key
                break
        else:
            raise RuntimeError(
                "No independent capture stream available; reduce active graph entries"
            )
        self.logits = torch.empty_like(logits)
        self.logits.copy_(logits)
        self.info = SimpleNamespace(
            temperatures=torch.ones(
                (self.bs, 1), device=logits.device, dtype=torch.float32
            ),
            top_ks=torch.full((self.bs,), 20, device=logits.device, dtype=torch.int32),
            top_ps=torch.full(
                (self.bs,), 0.95, device=logits.device, dtype=torch.float32
            ),
            min_ps=torch.zeros((self.bs,), device=logits.device, dtype=torch.float32),
            need_top_k_sampling=True,
            need_top_p_sampling=True,
            need_min_p_sampling=False,
        )
        self.fi_buffers = {}
        self.fi_retired = []
        self.workspace_frozen = False
        self.native_key = (
            self.bs * 16,
            str(logits.device),
            self.thread,
            int(self.capture_stream.cuda_stream),
        )
        self.native_buffers = None
        self.graph = torch.cuda.CUDAGraph()
        self.bundle = self.output = None
        _ACTIVE_CAPTURE_STREAMS.add(self.stream_key)

    def _workspace(self, name, num_bytes, device, zero_init=False):
        key = (name, str(device))
        buf = self.fi_buffers.get(key)
        if buf is None or buf.numel() < num_bytes:
            if self.workspace_frozen:
                raise RuntimeError(
                    "Unexpected FI scratch growth during capture: " + name
                )
            if buf is not None:
                self.fi_retired.append(buf)
            fn = self.cache.torch.zeros if zero_init else self.cache.torch.empty
            buf = fn(num_bytes, device=device, dtype=self.cache.torch.uint8)
            self.fi_buffers[key] = buf
        return buf

    @contextmanager
    def _private_fi_workspaces(self):
        modules = [
            importlib.import_module("flashinfer.topk"),
            importlib.import_module("flashinfer.sampling"),
        ]
        originals = {module: module._get_cache_buf for module in modules}
        owner = self.thread
        capture = int(self.capture_stream.cuda_stream)

        def wrap(original):

            def get(name, num_bytes, device, zero_init=False):
                if (
                    threading.get_ident() == owner
                    and int(self.cache.torch.cuda.current_stream(device).cuda_stream)
                    == capture
                ):
                    return self._workspace(name, num_bytes, device, zero_init)
                return original(name, num_bytes, device, zero_init=zero_init)

            return get

        try:
            for module, original in originals.items():
                module._get_cache_buf = wrap(original)
            yield
        finally:
            for module, original in originals.items():
                module._get_cache_buf = original

    def _produce(self):
        b = self.cache.producer.build_sparse_joint(
            logits=self.logits,
            info=self.info,
            bs=self.bs,
            width=16,
            native=self.cache.native,
        )
        return (
            b,
            self.cache.materializer.materialize(self.cache.torch, b)
            if self.mode == "dense"
            else b,
        )

    def prepare(self):
        torch = self.cache.torch
        self.capture_stream.wait_stream(self.caller_stream)
        with self._private_fi_workspaces():
            with torch.cuda.stream(self.capture_stream):
                for _ in range(self.cache.warmup):
                    warm = self._produce()
                self.capture_stream.synchronize()
                del warm
            self.native_buffers = self.cache.producer._buffers[self.native_key]
            self.workspace_frozen = True
            with torch.cuda.graph(
                self.graph,
                stream=self.capture_stream,
                capture_error_mode="thread_local",
            ):
                (self.bundle, self.output) = self._produce()
        assert self.cache.producer._buffers.get(self.native_key) is self.native_buffers
        self.cache.producer._buffers.pop(self.native_key)
        self.graph.replay()
        self.caller_stream.synchronize()


class GraphP4Cache:
    def __init__(self, torch, producer, materializer, native):
        (self.torch, self.producer, self.materializer, self.native) = (
            torch,
            producer,
            materializer,
            native,
        )
        (self.diagnostic, self.warmup) = (False, 4)
        self.entries = {}

    def key(self, logits, mode):
        assert mode in ("sparse", "dense") and logits.is_cuda
        assert logits.dtype == self.torch.float32 and logits.is_contiguous()
        assert (
            logits.ndim == 2
            and logits.shape[1] == 248320
            and (logits.shape[0] % 16 == 0)
        )
        bs = logits.shape[0] // 16
        assert 1 <= bs <= 8
        return (
            bs,
            str(logits.device),
            int(self.torch.cuda.current_stream(logits.device).cuda_stream),
            threading.get_ident(),
            mode,
        )

    def prepare(self, logits, mode):
        key = self.key(logits, mode)
        if key not in self.entries:
            with _PREPARE_LOCK:
                entry = _Entry(self, logits, key)
                try:
                    entry.prepare()
                except BaseException:
                    try:
                        entry.capture_stream.synchronize()
                    except BaseException:
                        pass
                    finally:
                        current = self.producer._buffers.get(entry.native_key)
                        if (
                            entry.native_buffers is None
                            or current is entry.native_buffers
                        ):
                            self.producer._buffers.pop(entry.native_key, None)
                        _ACTIVE_CAPTURE_STREAMS.discard(entry.stream_key)
                    raise
                self.entries[key] = entry
        return self.entries[key]

    def run(self, logits, mode):
        entry = self.entries.get(self.key(logits, mode))
        if entry is None:
            raise RuntimeError("Graph entry must be prepared outside measurement")
        entry.logits.copy_(logits)
        entry.graph.replay()
        return entry.output

    def close(self):
        """Explicit owner cleanup outside measurement; borrowed outputs expire."""
        devices = {entry.device for entry in self.entries.values()}
        for device in devices:
            self.torch.cuda.synchronize(device)
        for entry in self.entries.values():
            if self.producer._buffers.get(entry.native_key) is entry.native_buffers:
                self.producer._buffers.pop(entry.native_key, None)
            _ACTIVE_CAPTURE_STREAMS.discard(entry.stream_key)
            entry.graph.reset()
            entry.output = entry.bundle = entry.native_buffers = None
            entry.fi_buffers.clear()
            entry.fi_retired.clear()
            entry.logits = entry.info = None
        self.entries.clear()
