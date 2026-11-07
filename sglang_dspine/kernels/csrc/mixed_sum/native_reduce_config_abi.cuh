// ReduceConfig declaration and inline helpers from PyTorch v2.14.1 Reduce.cuh.
// BSD-3-Clause; see THIRD_PARTY_NOTICES.txt for the upstream notice.
// Source: https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/cuda/Reduce.cuh
// Modified for DSpine: extract the declaration and inline helpers for the pinned ABI.
// Calls the matching libtorch_cuda.so setReduceConfig<float,float,4,4>.
#pragma once
#include <ATen/native/TensorIterator.h>
#include <c10/macros/Macros.h>
#include <c10/util/complex.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
namespace at::cuda {
cudaDeviceProp* getCurrentDeviceProperties();
int warp_size();
}
namespace at::native {
static inline int64_t div_up(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

// returns floor(log2(n))
static inline int last_pow2(int n) {
  n |= (n >>  1);
  n |= (n >>  2);
  n |= (n >>  4);
  n |= (n >>  8);
  n |= (n >> 16);
  return std::max(1, n - (n >> 1));
}

template <typename T>
struct mnt_wrapper {
  static constexpr int MAX_NUM_THREADS = 512;
};

template <>
struct mnt_wrapper <c10::complex<double>>{
  static constexpr int MAX_NUM_THREADS = 256;
};

constexpr int max_reduce_threads(c10::ScalarType type) {
  return type == kComplexDouble ? 256 : 512;
}

struct ReduceConfig {
  static constexpr int BLOCK_X = 0;
  static constexpr int BLOCK_Y = 1;
  static constexpr int CTA = 2;

  ReduceConfig(int element_size_bytes, int num_outputs, int num_inputs)
    : element_size_bytes(element_size_bytes)
    , num_inputs(num_inputs)
    , num_outputs(num_outputs) {}
  int element_size_bytes;
  int num_inputs;
  int num_outputs;
  int step_input = 1;
  int step_output = 1;
  int ctas_per_output = 1;
  int input_mult[3] = {0, 0, 0};
  int output_mult[2] = {0, 0};

  int block_width;
  int block_height;
  int num_threads;

  bool vectorize_input = false;
  int output_vec_size = 1;

  template <typename T>
  void set_block_dimension(int64_t dim0, int64_t dim1) {
    const int max_num_threads = mnt_wrapper<T>::MAX_NUM_THREADS / output_vec_size;
    int dim0_pow2 = dim0 < max_num_threads ? static_cast<int>(last_pow2(dim0)) : max_num_threads;
    int dim1_pow2 = dim1 < max_num_threads ? static_cast<int>(last_pow2(dim1)) : max_num_threads;
    block_width = std::min(dim0_pow2, int(at::cuda::warp_size()));
    block_height = std::min(dim1_pow2, int(max_num_threads / block_width));
    block_width = std::min(dim0_pow2, int(max_num_threads / block_height));
    num_threads = block_width * block_height;
  }

  int split_input(int parallelism) {
    int step = step_input;
    step_input *= parallelism;
    return step;
  }

  int split_output(int parallelism) {
    int step = step_output;
    step_output *= parallelism;
    return step;
  }

  dim3 block() const {
    return dim3(block_width, block_height);
  }

  dim3 grid() const {
    return dim3(div_up(num_outputs / output_vec_size, step_output), ctas_per_output);
  }

  C10_HOST_DEVICE bool should_block_x_reduce() const {
    return input_mult[BLOCK_X] != 0;
  }

  C10_HOST_DEVICE bool should_block_y_reduce() const {
    return input_mult[BLOCK_Y] != 0;
  }

  C10_HOST_DEVICE bool should_global_reduce() const {
    return input_mult[CTA] != 0;
  }

  C10_DEVICE bool should_store(int output_idx) const {
    return output_idx < num_outputs &&
      (!should_block_x_reduce() || threadIdx.x == 0) &&
      (!should_block_y_reduce() || threadIdx.y == 0);
  }

  C10_DEVICE bool should_reduce_tail() const {
    return (!should_block_y_reduce() || threadIdx.y == 0) &&
      (!should_global_reduce() || blockIdx.y == 0);
  }

  C10_HOST_DEVICE int input_idx() const {
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int cta2 = blockIdx.y;
    return (lane * input_mult[BLOCK_X] +
            warp * input_mult[BLOCK_Y] +
            cta2 * input_mult[CTA]);
  }

  template <int output_vec_size>
  C10_HOST_DEVICE int output_idx() const {
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int cta1 = blockIdx.x;
    return (lane * output_mult[BLOCK_X] +
            warp * output_mult[BLOCK_Y] +
            cta1 * step_output) * output_vec_size;
  }

  C10_DEVICE int shared_memory_offset(int offset) const {
    return threadIdx.x + (threadIdx.y + offset) * blockDim.x;
  }

  C10_DEVICE int staging_memory_offset(int cta2) const {
    int offset = cta2 + blockIdx.x * gridDim.y;
    if (!should_block_x_reduce()) {
      offset = threadIdx.x + offset * blockDim.x;
    }
    return offset;
  }

  int shared_memory_size() const {
    if (!should_block_y_reduce() &&
        (!should_block_x_reduce() ||
         block_width <= at::cuda::warp_size())) {
      return 0;
    }
    return element_size_bytes * num_threads * output_vec_size;
  }

  int64_t global_memory_size() const {
    if (!should_global_reduce()) {
      return 0;
    }
    auto size = (int64_t)element_size_bytes * num_outputs * ctas_per_output;
    if (!should_block_x_reduce()) {
      size *= block().x * output_vec_size;
    }
    return size;
  }

  int semaphore_size() const {
    if (!should_global_reduce()) {
      return 0;
    }
    return sizeof(int) * grid().x;
  }

  int values_per_thread() const {
    return div_up(num_inputs, step_input);
  }

  int mock_values_per_thread(int parallelism) {
    return div_up(num_inputs, step_input * parallelism);
  }
};


static_assert(sizeof(ReduceConfig) == 64);
static_assert(offsetof(ReduceConfig, block_width) == 44);
static_assert(offsetof(ReduceConfig, vectorize_input) == 56);
static_assert(offsetof(ReduceConfig, output_vec_size) == 60);
static_assert(std::is_trivially_copyable_v<ReduceConfig>);
template<typename arg_t, typename scalar_t, int vt0, int input_vec_size=vt0>
ReduceConfig setReduceConfig(const TensorIterator& iter);
} // namespace at::native
