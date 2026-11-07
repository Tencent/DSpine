// Sparse and dense FP32 sums matching the pinned PyTorch reduction order.
// Uses PyTorch 2.14.1 ReduceConfig geometry for SM90 GPUs with 132 SMs.
#include <ATen/ATen.h>
#include "native_reduce_config_abi.cuh"
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <vector>

namespace {
constexpr int V = 248320;

int ctas_for_rows(int rows) {
  const auto* p = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(p->major == 9 && p->minor == 0 && p->multiProcessorCount == 132 &&
              p->maxThreadsPerMultiProcessor == 2048,
              "Unsupported hardware geometry for the exact sparse reduction");
  TORCH_CHECK(rows >= 16 && rows <= 128 && rows % 16 == 0,
              "Unsupported row geometry for the exact sparse reduction");
  const int c1 = (528 + rows - 1) / rows;
  return std::min(31, std::max(2, c1));
}

void float_cuda(const at::Tensor& t) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kFloat && t.is_contiguous(),
              "Expected contiguous CUDA float32 tensor");
}

// Each native (CTA,tid,vec-lane) gets identical nonzero additions, in original
// increasing vector index. Skipping +0 is exact for finite nonnegative normal p.
__global__ void mixed_partials(const int* ids, const float* weights,
                                const int* lengths, const bool* use_sparse,
                                const float* dense, float* partials,
                                int capacity, int ctas) {
  const int row = blockIdx.x, cta = blockIdx.y;
  const int tx = threadIdx.x, ty = threadIdx.y, tid = tx + 32 * ty;
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  if (use_sparse[row]) {
    const int n = lengths[row];
    for (int j = 0; j < n; ++j) {
    const int k = ids[row * capacity + j];
    const int vec = k >> 2;
    if ((vec & 511) == tid && (vec / 512) % ctas == cta) {
      const float w = weights[row * capacity + j];
      switch (k & 3) {
        case 0: a0 = __fadd_rn(a0, w); break;
        case 1: a1 = __fadd_rn(a1, w); break;
        case 2: a2 = __fadd_rn(a2, w); break;
        case 3: a3 = __fadd_rn(a3, w); break;
      }
    }
    }
  } else {
    // Literal original aligned vec4 thread traversal, including all zeros and
    // subnormals. Volatile PTX keeps the loads inside this device branch: safe
    // sparse rows may have an unwritten/NaN-poisoned dense row allocation.
    int vec = tid + 512 * cta;
    const float* row_data = dense + static_cast<int64_t>(row) * V;
    for (; vec < V / 4; vec += 512 * ctas) {
      float x0, x1, x2, x3;
      const float* ptr = row_data + vec * 4;
      asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
                   : "=f"(x0), "=f"(x1), "=f"(x2), "=f"(x3)
                   : "l"(ptr) : "memory");
      a0 = __fadd_rn(a0, x0);
      a1 = __fadd_rn(a1, x1);
      a2 = __fadd_rn(a2, x2);
      a3 = __fadd_rn(a3, x3);
    }
  }
  float value = __fadd_rn(__fadd_rn(__fadd_rn(a0, a1), a2), a3);
  #pragma unroll
  for (int delta = 16; delta; delta >>= 1) {
    value = __fadd_rn(value, __shfl_down_sync(0xffffffff, value, delta));
  }
  __shared__ float warps[16];
  if (tx == 0) warps[ty] = value;
  #pragma unroll
  for (int delta = 8; delta; delta >>= 1) {
    __syncthreads();
    if (tx == 0 && ty < delta) {
      value = __fadd_rn(value, warps[ty + delta]);
      warps[ty] = value;
    }
  }
  if (tid == 0) partials[row * ctas + cta] = value;
}

__global__ void finish_partials(const float* partials, float* out, int ctas) {
  const int row = blockIdx.x, lane = threadIdx.x;
  // Original global-reduce block starts from ident+partial, then adds 15
  // all-zero warps in Y before this X reduction. Both are bit-preserving here.
  float value = lane < ctas ? __fadd_rn(0.0f, partials[row * ctas + lane]) : 0.0f;
  #pragma unroll
  for (int delta = 16; delta; delta >>= 1) {
    value = __fadd_rn(value, __shfl_down_sync(0xffffffff, value, delta));
  }
  if (lane == 0) out[row] = value;
}
}  // namespace

std::vector<int64_t> native_geometry_cuda(const at::Tensor& dense, at::Tensor out) {
  float_cuda(dense);
  float_cuda(out);
  TORCH_CHECK(dense.dim() == 2 && dense.size(1) == V && out.dim() == 2 &&
              out.size(0) == dense.size(0) && out.size(1) == 1,
              "Query requires [16*BS,V248320] and [16*BS,1]");
  TORCH_CHECK(dense.device() == out.device() &&
              reinterpret_cast<uintptr_t>(dense.data_ptr<float>()) % 16 == 0,
              "Original dense reduction must be 16-byte aligned");
  c10::cuda::CUDAGuard guard(dense.device());
  const int expected = ctas_for_rows(dense.size(0));
  auto iter = at::TensorIterator::reduce_op(out, dense);
  const auto cfg = at::native::setReduceConfig<float, float, 4, 4>(iter);
  TORCH_CHECK(cfg.block_width == 32 && cfg.block_height == 16 &&
              cfg.ctas_per_output == expected && cfg.vectorize_input &&
              cfg.output_vec_size == 1 && cfg.input_mult[0] == 1 &&
              cfg.input_mult[1] == 32 && cfg.input_mult[2] == 512 &&
              cfg.step_input == expected * 512,
              "Actual installed ATen geometry differs from proved sparse mapping");
  return {cfg.block_width, cfg.block_height, cfg.ctas_per_output, cfg.step_input,
          cfg.input_mult[0], cfg.input_mult[1], cfg.input_mult[2],
          static_cast<int64_t>(cfg.vectorize_input), cfg.output_vec_size};
}

void mixed_sum_cuda(const at::Tensor& ids, const at::Tensor& weights,
                    const at::Tensor& lengths, const at::Tensor& use_sparse,
                    const at::Tensor& dense, at::Tensor partials, at::Tensor out) {
  float_cuda(weights);
  float_cuda(partials);
  float_cuda(out);
  float_cuda(dense);
  TORCH_CHECK(ids.is_cuda() && ids.scalar_type() == at::kInt && ids.is_contiguous() &&
              lengths.is_cuda() && lengths.scalar_type() == at::kInt && lengths.is_contiguous(),
              "IDs/lengths must be contiguous int32 CUDA");
  TORCH_CHECK(ids.dim() == 2 && ids.sizes() == weights.sizes() &&
              ids.size(1) >= 1 && ids.size(1) <= 32 && lengths.dim() == 1 &&
              lengths.size(0) == ids.size(0) && out.numel() == ids.size(0),
              "Invalid sparse shapes");
  TORCH_CHECK(use_sparse.is_cuda() && use_sparse.scalar_type() == at::kBool &&
              use_sparse.is_contiguous() && use_sparse.numel() == ids.size(0) &&
              dense.dim() == 2 && dense.size(0) == ids.size(0) && dense.size(1) == V &&
              reinterpret_cast<uintptr_t>(dense.data_ptr<float>()) % 16 == 0,
              "Expected device per-row bool and original aligned [R,V] dense fallback");
  TORCH_CHECK(ids.device() == weights.device() && ids.device() == lengths.device() &&
              ids.device() == partials.device() && ids.device() == out.device() &&
              ids.device() == use_sparse.device() && ids.device() == dense.device(),
              "All tensors must be on the same device");
  c10::cuda::CUDAGuard guard(ids.device());
  const int rows = ids.size(0), ctas = ctas_for_rows(rows);
  TORCH_CHECK(partials.numel() >= rows * ctas, "Insufficient partial workspace");
  // Packed support is constructed by the sparse producer; uncertain rows
  // retain the original dense fallback.
  const auto stream = c10::cuda::getCurrentCUDAStream(ids.get_device());
  mixed_partials<<<dim3(rows, ctas), dim3(32, 16), 0, stream>>>(
      ids.data_ptr<int>(), weights.data_ptr<float>(), lengths.data_ptr<int>(),
      use_sparse.data_ptr<bool>(), dense.data_ptr<float>(), partials.data_ptr<float>(),
      ids.size(1), ctas);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  finish_partials<<<rows, 32, 0, stream>>>(partials.data_ptr<float>(), out.data_ptr<float>(), ctas);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
