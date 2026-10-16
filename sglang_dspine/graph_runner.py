"""Run DSpine's eager sampled refinement after draft CUDA-graph replay.

Sampled rounds must update ``out`` before DFlashWorkerV2 reads the proposal.
The plugin selects this runner only for the DSPINE draft model.
"""

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
