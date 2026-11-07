#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <flashinfer/math.cuh>
#include <flashinfer/topk.cuh>
#include "conditional_topk.cuh"

void conditional_topk_out(at::Tensor probs,at::Tensor ks,at::Tensor safe,
 at::Tensor output,at::Tensor states,at::Tensor visits,bool track) {
  TORCH_CHECK(probs.is_cuda() && probs.scalar_type()==at::kFloat && probs.dim()==2 && probs.is_contiguous());
  TORCH_CHECK(ks.is_cuda() && ks.scalar_type()==at::kInt && ks.is_contiguous() && ks.numel()==probs.size(0));
  TORCH_CHECK(safe.is_cuda() && safe.scalar_type()==at::kBool && safe.is_contiguous() && safe.numel()==probs.size(0));
  TORCH_CHECK(output.is_cuda() && output.scalar_type()==at::kFloat && output.is_contiguous() && output.sizes()==probs.sizes());
  TORCH_CHECK(states.is_cuda() && states.scalar_type()==at::kByte && states.is_contiguous());
  TORCH_CHECK(visits.is_cuda() && visits.scalar_type()==at::kInt && visits.is_contiguous() && visits.numel()>=probs.size(0));
  for(auto const& tensor:{ks,safe,output,states,visits})TORCH_CHECK(tensor.device()==probs.device());
  static_assert(sizeof(flashinfer::sampling::RadixRowState)<=4096);
  TORCH_CHECK(states.numel()>=probs.size(0)*4096,"Insufficient zero-initialized persistent states");
  c10::cuda::CUDAGuard guard(probs.device());
  auto stream=c10::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK((flashinfer::sampling::ConditionalRadixTopKRenormProbMultiCTA<float,int>(
    probs.data_ptr<float>(),output.data_ptr<float>(),ks.data_ptr<int>(),probs.size(0),0,probs.size(1),
    reinterpret_cast<flashinfer::sampling::RadixRowState*>(states.data_ptr<uint8_t>()),false,stream,
    safe.data_ptr<bool>(),visits.data_ptr<int>(),track)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
