#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "air_top_p_original.cuh"
#include "conditional_kernels.cuh"
namespace air = flashinfer::sampling::air_top_p;
static_assert(sizeof(air::Counter<float>) == 384);

__global__ void SparseAirCutApply(float const* probs, bool* support,
 air::Counter<float>* counters, bool const* certified, float const* cuts,
 int* visits, bool track, int vocabSize, int const* modes, float* p_threshold) {
  int const batchId=blockIdx.x;
  int const tid=threadIdx.x;
  if (certified[batchId]) {
    if (tid==0) p_threshold[batchId]=cuts[batchId];
    return;
  }
  float const threshold=air::twiddleOut<float>(counters[batchId].kthValueBits,false);
  if (tid==0) p_threshold[batchId]=threshold;
  if (modes[batchId]!=0) return;
  if (track && tid==0) visits[batchId*3+2]=1;
  // Exact original apply body remains for every numerically unsafe/native row.
  float threadSum=0.0f;
  for (int i=tid;i<vocabSize;i+=blockDim.x) {
    float val=probs[batchId*vocabSize+i];
    if (val>=threshold) threadSum+=static_cast<float>(val);
  }
  typedef cub::BlockReduce<float,1024> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp;
  float totalSum=BlockReduce(temp).Sum(threadSum);
  __shared__ float sharedNorm;
  if (tid==0) sharedNorm=(totalSum>1e-8f)?(1.0f/totalSum):1.0f;
  __syncthreads();
  float norm=sharedNorm;
  for (int i=tid;i<vocabSize;i+=blockDim.x) {
    float val=probs[batchId*vocabSize+i];
    float result=(val>=threshold)?static_cast<float>(val*norm):0.0f;
    support[batchId*vocabSize+i]=result>0.0f;
  }
}

void sparse_air_cut_out(at::Tensor probs, at::Tensor ps, at::Tensor certified,
 at::Tensor cuts, at::Tensor modes, at::Tensor p_threshold, at::Tensor support, at::Tensor workspace, at::Tensor visits, bool track) {
  TORCH_CHECK(probs.is_cuda() && probs.scalar_type()==at::kFloat && probs.dim()==2 && probs.is_contiguous());
  TORCH_CHECK(ps.is_cuda() && ps.scalar_type()==at::kFloat && ps.is_contiguous());
  TORCH_CHECK(certified.is_cuda() && certified.scalar_type()==at::kBool && certified.is_contiguous());
  TORCH_CHECK(cuts.is_cuda() && cuts.scalar_type()==at::kFloat && cuts.is_contiguous());
  TORCH_CHECK(support.is_cuda() && support.scalar_type()==at::kBool && support.sizes()==probs.sizes() && support.is_contiguous());
  TORCH_CHECK(workspace.is_cuda() && workspace.scalar_type()==at::kByte && workspace.is_contiguous());
  TORCH_CHECK(visits.is_cuda() && visits.scalar_type()==at::kInt && visits.is_contiguous());
  for (auto const& tensor:{ps,certified,cuts,support,workspace,visits}) TORCH_CHECK(tensor.device()==probs.device());
  TORCH_CHECK(modes.is_cuda() && modes.scalar_type()==at::kInt && modes.is_contiguous() && modes.numel()==probs.size(0));
  TORCH_CHECK(p_threshold.is_cuda() && p_threshold.scalar_type()==at::kFloat && p_threshold.is_contiguous() && p_threshold.numel()==probs.size(0));
  TORCH_CHECK(modes.device()==probs.device() && p_threshold.device()==probs.device());
  int batch=probs.size(0),vocab=probs.size(1);
  TORCH_CHECK(vocab>=air::NUM_BUCKETS && batch>0 && ps.numel()==batch && certified.numel()==batch && cuts.numel()==batch && visits.numel()>=batch*3);
  c10::cuda::CUDAGuard guard(probs.device());
  using HisT=air::HisT<false,float>;
  auto align=[](size_t x){return ((x+255)/256)*256;};
  auto bufLen=air::calcBufLen<float>(vocab);
  size_t cs=align(sizeof(air::Counter<float>)*batch), hs=align(sizeof(HisT)*air::NUM_BUCKETS*batch);
  size_t ns=align(sizeof(air::IdxT)*air::NUM_BUCKETS*batch), bs=align(sizeof(float)*bufLen*batch);
  TORCH_CHECK(static_cast<size_t>(workspace.numel())>=cs+hs+ns+2*bs);
  auto ws=workspace.data_ptr<uint8_t>();auto counters=reinterpret_cast<air::Counter<float>*>(ws);
  auto hist=reinterpret_cast<HisT*>(ws+cs);auto cnt=reinterpret_cast<air::IdxT*>(ws+cs+hs);
  auto b1=reinterpret_cast<float*>(ws+cs+hs+ns);auto b2=reinterpret_cast<float*>(ws+cs+hs+ns+bs);
  int device,sm;
  C10_CUDA_CHECK(cudaGetDevice(&device));C10_CUDA_CHECK(cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,device));
  // Deliberately use the ORIGINAL kernel occupancy and ORIGINAL full batch size.
  uint32_t blockNum=air::CalcAirTopPBlockNum<false,float>(batch,vocab,sm);
  auto stream=c10::cuda::getCurrentCUDAStream();
  air::ConditionalAirTopPRenormInitKernel<float,HisT><<<batch,256,0,stream>>>(
    counters,vocab,probs.data_ptr<float>(),ps.data_ptr<float>(),0.0f,hist,cnt,
    certified.data_ptr<bool>(),visits.data_ptr<int>(),track);
  dim3 grid(blockNum,batch);
  for (int pass=0;pass<air::NUM_PASSES<float>;++pass) {
    air::ConditionalAirTopPRenormRadixKernel<false,float><<<grid,air::BLOCK_SIZE,0,stream>>>(
      counters,hist,cnt,pass,b1,b2,certified.data_ptr<bool>(),visits.data_ptr<int>(),track);
  }
  SparseAirCutApply<<<batch,1024,0,stream>>>(probs.data_ptr<float>(),support.data_ptr<bool>(),counters,
    certified.data_ptr<bool>(),cuts.data_ptr<float>(),visits.data_ptr<int>(),track,vocab,modes.data_ptr<int>(),p_threshold.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
