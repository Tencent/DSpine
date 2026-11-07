"""DSpine serving on SGLang 0.5.22."""

from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)


class DSpineDraftCudaGraphRunner(DecodeCudaGraphRunner):
    def execute(self, forward_batch, pp_proxy_tensors=None):
        output = super().execute(forward_batch, pp_proxy_tensors=pp_proxy_tensors)
        sampler = self.model_runner.model.dspine_sampler
        if sampler is None:
            raise RuntimeError("DSPINE draft graph replayed before its sampler was set")
        sampler.after_graph_replay(forward_batch)
        return output
