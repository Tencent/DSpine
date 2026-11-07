/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Source: https://github.com/flashinfer-ai/flashinfer/blob/v0.7.1/include/flashinfer/topk.cuh
// Modified for DSpine: add per-row dispatch guards and visit counters.

namespace flashinfer { namespace sampling {
template <uint32_t BLOCK_THREADS, uint32_t VEC_SIZE, bool SINGLE_CTA, bool DETERMINISTIC,
          typename DType, typename IdType>
__global__ void __launch_bounds__(BLOCK_THREADS) ConditionalRadixTopKRenormProbKernel_MultiCTA(
    DType* probs,          // [batch, vocab_size]
    DType* renormed_prob,  // [batch, vocab_size]
    IdType* top_k_arr,     // [batch] or nullptr
    uint32_t top_k_val, uint32_t vocab_size, uint32_t batch_size,
    RadixRowState* row_states,  // [num_groups] (nullptr if SINGLE_CTA)
    uint32_t chunk_size,        // elements per CTA
    uint32_t ctas_per_group, bool const* k_safe, int* visits, bool track)    // CTAs per row (1 if SINGLE_CTA)
{
  using Traits = RadixTopKTraits<DType>;
  using OrderedType = typename Traits::OrderedType;

  constexpr uint32_t RADIX = 256;  // 8-bit radix

  const uint32_t global_cta_id = blockIdx.x;
  const uint32_t group_id = global_cta_id / ctas_per_group;
  const uint32_t cta_in_group = global_cta_id % ctas_per_group;
  const uint32_t tx = threadIdx.x;

  // Shared memory layout: [fixed storage] [ordered values cache]
  extern __shared__ uint8_t smem[];

  // Fixed shared memory (at the beginning)
  // histogram[256] + suffix[256] + scalars[4] + sum_local[1]
  constexpr size_t fixed_smem_size = sizeof(uint32_t) * (RADIX + RADIX + 4) + sizeof(float);
  uint32_t* local_histogram = reinterpret_cast<uint32_t*>(smem);
  uint32_t* suffix_sum = local_histogram + RADIX;
  uint32_t* shared_scalars = suffix_sum + RADIX;
  float* shared_sum = reinterpret_cast<float*>(shared_scalars + 4);

  // Align ordered values cache to 16 bytes
  size_t ordered_offset = ((fixed_smem_size + 15) / 16) * 16;
  OrderedType* shared_ordered = reinterpret_cast<OrderedType*>(smem + ordered_offset);

  // State pointer only used when not SINGLE_CTA
  RadixRowState* state = nullptr;
  if constexpr (!SINGLE_CTA) {
    state = &row_states[group_id];
  }

  // Calculate total number of iterations for persistent loop
  uint32_t num_groups = gridDim.x / ctas_per_group;
  uint32_t total_iterations = (batch_size + num_groups - 1) / num_groups;

  int barrier_phase = 0;
  uint32_t native_iter = 0;

  // Persistent loop over rows
  for (uint32_t iter = 0; iter < total_iterations; iter++) {
    uint32_t row_idx = group_id + iter * num_groups;

    if (row_idx >= batch_size) break;
    if (k_safe[row_idx]) continue;
    {
    const uint32_t iter = native_iter++;
    if (track && tx == 0) atomicAdd(visits + row_idx, 1);

    const uint32_t chunk_start = cta_in_group * chunk_size;
    const uint32_t chunk_end = min(chunk_start + chunk_size, vocab_size);
    const uint32_t actual_chunk_size = chunk_end - chunk_start;

    uint32_t k = top_k_arr == nullptr ? top_k_val : top_k_arr[row_idx];

    // For RenormProb, pivot is compared with probs (must be non-negative)
    DType pivot = DType(0);
    float normalizer = 1.0f;

    if (k >= vocab_size) {
      // k >= vocab_size: no filtering needed, just compute sum and renormalize
      // Stage 1: Compute sum
      float thread_sum = 0.0f;
      vec_t<DType, VEC_SIZE> data_vec;
      const uint32_t aligned_size = (actual_chunk_size / VEC_SIZE) * VEC_SIZE;

#pragma unroll 2
      for (uint32_t i = tx * VEC_SIZE; i < aligned_size; i += BLOCK_THREADS * VEC_SIZE) {
        data_vec.cast_load(probs + row_idx * vocab_size + chunk_start + i);
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
          thread_sum += float(data_vec[j]);
        }
      }
      // Handle tail
      for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK_THREADS) {
        thread_sum += float(probs[row_idx * vocab_size + chunk_start + i]);
      }

      // Block reduction for sum
      typedef cub::BlockReduce<float, BLOCK_THREADS> BlockReduce;
      __shared__ typename BlockReduce::TempStorage temp_storage;
      float block_sum = BlockReduce(temp_storage).Sum(thread_sum);
      __syncthreads();

      if constexpr (!SINGLE_CTA) {
        // Multi-CTA: accumulate the group sum. The deterministic path uses
        // fixed-point integer atomics (order-independent); the default path
        // keeps float atomicAdd, whose CTA-arrival-order-dependent rounding
        // makes results vary across invocations.
        if (tx == 0) {
          if (cta_in_group == 0) {
            if constexpr (DETERMINISTIC) {
              state->sum_topk_fixed = 0;  // First CTA initializes
            } else {
              state->sum_topk = 0.0f;  // First CTA initializes
            }
          }
        }
        // Barrier for initialization
        AdvanceRadixGroupBarrier(state, barrier_phase, ctas_per_group, tx);

        if (tx == 0 && block_sum > 0) {
          if constexpr (DETERMINISTIC) {
            atomicAdd(&state->sum_topk_fixed,
                      static_cast<unsigned long long>(
                          double(block_sum) * kRenormDeterministicScale + 0.5));
          } else {
            atomicAdd(&state->sum_topk, block_sum);
          }
        }

        // Barrier to ensure all CTAs have contributed
        AdvanceRadixGroupBarrier(state, barrier_phase, ctas_per_group, tx);
        float group_sum;
        if constexpr (DETERMINISTIC) {
          group_sum = float(double(state->sum_topk_fixed) / kRenormDeterministicScale);
        } else {
          group_sum = state->sum_topk;
        }
        normalizer = math::ptx_rcp(max(group_sum, 1e-8f));
      } else {
        // Single-CTA: use block_sum directly
        if (tx == 0) {
          *shared_sum = block_sum;
        }
        __syncthreads();
        normalizer = math::ptx_rcp(max(*shared_sum, 1e-8f));
      }

      // Normalize and store
#pragma unroll 2
      for (uint32_t i = tx * VEC_SIZE; i < aligned_size; i += BLOCK_THREADS * VEC_SIZE) {
        data_vec.cast_load(probs + row_idx * vocab_size + chunk_start + i);
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
          data_vec[j] = DType(float(data_vec[j]) * normalizer);
        }
        data_vec.store(renormed_prob + row_idx * vocab_size + chunk_start + i);
      }
      for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK_THREADS) {
        renormed_prob[row_idx * vocab_size + chunk_start + i] =
            DType(float(probs[row_idx * vocab_size + chunk_start + i]) * normalizer);
      }

      // Clear histogram for next iteration (in case it's k < vocab_size)
      // Only needed for multi-CTA mode; single-CTA uses shared memory cleared each iteration
      // Next iteration (iter+1) will use histogram[((iter+1)*NUM_ROUNDS) % 3] for its first round
      if constexpr (!SINGLE_CTA) {
        constexpr uint32_t NUM_ROUNDS = sizeof(OrderedType) * 8 / 8;  // ORDERED_BITS / RADIX_BITS
        uint32_t next_first_hist_idx = ((iter + 1) * NUM_ROUNDS) % 3;
        if (cta_in_group == 0) {
          for (uint32_t i = tx; i < RADIX; i += BLOCK_THREADS) {
            state->histogram[next_first_hist_idx][i] = 0;
          }
        }
        // No sync needed - next iteration's barrier will ensure visibility
      }
      continue;
    }

    // ========== Stage 1: Find pivot ==========
    uint32_t local_gt_count = 0;  // Not used in this kernel
    uint32_t local_eq_count = 0;  // Not used in this kernel
    auto ordered_pivot = RadixSelectFindPivot<BLOCK_THREADS, VEC_SIZE, SINGLE_CTA, false, DType>(
        probs + row_idx * vocab_size, shared_ordered, local_histogram, suffix_sum, shared_scalars,
        state, chunk_start, actual_chunk_size, k, barrier_phase, ctas_per_group, cta_in_group, tx,
        iter, local_gt_count, local_eq_count);
    pivot = Traits::FromOrdered(ordered_pivot);

    // ========== Stage 2: Compute sum of elements >= pivot ==========
    float thread_sum = 0.0f;
    vec_t<DType, VEC_SIZE> data_vec;
    const uint32_t aligned_size = (actual_chunk_size / VEC_SIZE) * VEC_SIZE;

#pragma unroll 2
    for (uint32_t i = tx * VEC_SIZE; i < aligned_size; i += BLOCK_THREADS * VEC_SIZE) {
      data_vec.cast_load(probs + row_idx * vocab_size + chunk_start + i);
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        if (data_vec[j] >= pivot) {
          thread_sum += float(data_vec[j]);
        }
      }
    }
    // Handle tail
    for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK_THREADS) {
      DType val = probs[row_idx * vocab_size + chunk_start + i];
      if (val >= pivot) {
        thread_sum += float(val);
      }
    }

    // Block reduction for sum
    typedef cub::BlockReduce<float, BLOCK_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float block_sum = BlockReduce(temp_storage).Sum(thread_sum);
    __syncthreads();

    if constexpr (!SINGLE_CTA) {
      // Multi-CTA: accumulate the group sum. See the k >= vocab_size branch
      // above for the deterministic-vs-float-atomic rationale.
      if (tx == 0) {
        if (cta_in_group == 0) {
          if constexpr (DETERMINISTIC) {
            state->sum_topk_fixed = 0;  // First CTA initializes
          } else {
            state->sum_topk = 0.0f;  // First CTA initializes
          }
        }
      }
      // Barrier for initialization
      AdvanceRadixGroupBarrier(state, barrier_phase, ctas_per_group, tx);

      if (tx == 0 && block_sum > 0) {
        if constexpr (DETERMINISTIC) {
          atomicAdd(
              &state->sum_topk_fixed,
              static_cast<unsigned long long>(double(block_sum) * kRenormDeterministicScale + 0.5));
        } else {
          atomicAdd(&state->sum_topk, block_sum);
        }
      }

      // Barrier to ensure all CTAs have contributed
      AdvanceRadixGroupBarrier(state, barrier_phase, ctas_per_group, tx);
      float group_sum;
      if constexpr (DETERMINISTIC) {
        group_sum = float(double(state->sum_topk_fixed) / kRenormDeterministicScale);
      } else {
        group_sum = state->sum_topk;
      }
      normalizer = math::ptx_rcp(max(group_sum, 1e-8f));
    } else {
      // Single-CTA: use block_sum directly
      if (tx == 0) {
        *shared_sum = block_sum;
      }
      __syncthreads();
      normalizer = math::ptx_rcp(max(*shared_sum, 1e-8f));
    }

    // ========== Stage 3: Normalize elements >= pivot, set others to 0 ==========
#pragma unroll 2
    for (uint32_t i = tx * VEC_SIZE; i < aligned_size; i += BLOCK_THREADS * VEC_SIZE) {
      data_vec.cast_load(probs + row_idx * vocab_size + chunk_start + i);
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        data_vec[j] = (data_vec[j] >= pivot) ? DType(float(data_vec[j]) * normalizer) : DType(0);
      }
      data_vec.store(renormed_prob + row_idx * vocab_size + chunk_start + i);
    }
    // Handle tail
    for (uint32_t i = aligned_size + tx; i < actual_chunk_size; i += BLOCK_THREADS) {
      DType val = probs[row_idx * vocab_size + chunk_start + i];
      renormed_prob[row_idx * vocab_size + chunk_start + i] =
          (val >= pivot) ? DType(float(val) * normalizer) : DType(0);
    }
    }
  }

  // Reset group state for the next launch, from the last CTA to exit (only for multi-CTA).
  // The leading CTA must not do this: it could zero the arrival counter while a peer CTA is
  // still spinning in its final wait_ge, wedging the stream forever (issue #3610).
  if constexpr (!SINGLE_CTA) {
    RadixGroupResetStateLastCTA<BLOCK_THREADS>(state, nullptr, barrier_phase, ctas_per_group, tx);
  }
}

template <typename DType, typename IdType>
cudaError_t ConditionalRadixTopKRenormProbMultiCTA(DType* probs, DType* renormed_prob, IdType* top_k_arr,
                                        uint32_t batch_size, uint32_t top_k_val,
                                        uint32_t vocab_size, RadixRowState* row_states_buffer,
                                        bool deterministic, cudaStream_t stream, bool const* k_safe, int* visits, bool track) {
  using OrderedType = typename RadixTopKTraits<DType>::OrderedType;
  constexpr uint32_t BLOCK_THREADS = 1024;
  const uint32_t vec_size = std::gcd(16 / sizeof(DType), vocab_size);

  // Get device properties
  int device;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&device));
  int num_sms;
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
  int max_smem_per_block;
  FLASHINFER_CUDA_CALL(
      cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));

  // Fixed shared memory overhead: histogram[256] + suffix_sum[256] + 4 scalars + 1 float
  constexpr size_t fixed_smem_size = sizeof(uint32_t) * (256 + 256 + 4) + sizeof(float);
  constexpr size_t fixed_smem_aligned = round_up(fixed_smem_size, 16);

  // Calculate max chunk size that fits in shared memory
  const size_t available_for_ordered = GetRadixTopKAvailableOrderedSmemBytes<BLOCK_THREADS>(
      max_smem_per_block, fixed_smem_aligned, false);
  if (available_for_ordered == 0) {
    return cudaErrorInvalidValue;
  }
  uint32_t max_chunk_elements = available_for_ordered / sizeof(OrderedType);
  max_chunk_elements = round_down(max_chunk_elements, vec_size);
  const uint32_t min_chunk_size = vec_size * BLOCK_THREADS;
  max_chunk_elements = std::max(max_chunk_elements, min_chunk_size);

  uint32_t ctas_per_group = ceil_div(vocab_size, max_chunk_elements);
  uint32_t chunk_size = ceil_div(vocab_size, ctas_per_group);
  chunk_size = round_up(chunk_size, vec_size);
  chunk_size = std::min(chunk_size, max_chunk_elements);

  const uint32_t smem_size = fixed_smem_aligned + chunk_size * sizeof(OrderedType);
  const bool single_cta = (ctas_per_group == 1);

  // Calculate number of groups (how many rows to process concurrently)
  uint32_t num_groups = std::min(static_cast<uint32_t>(num_sms) / ctas_per_group, batch_size);
  if (num_groups == 0) num_groups = 1;
  uint32_t total_ctas = num_groups * ctas_per_group;

  DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
    // Single-CTA sums with a block reduction in a fixed order, so it is
    // already deterministic; the DETERMINISTIC path only changes multi-CTA
    // accumulation.
    auto kernel =
        ConditionalRadixTopKRenormProbKernel_MultiCTA<BLOCK_THREADS, VEC_SIZE, true, false, DType, IdType>;
    if (!single_cta) {
      kernel = deterministic ? ConditionalRadixTopKRenormProbKernel_MultiCTA<BLOCK_THREADS, VEC_SIZE, false,
                                                                  true, DType, IdType>
                             : ConditionalRadixTopKRenormProbKernel_MultiCTA<BLOCK_THREADS, VEC_SIZE, false,
                                                                  false, DType, IdType>;
    }
    FLASHINFER_CUDA_CALL(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    dim3 nblks(total_ctas);
    dim3 nthrs(BLOCK_THREADS);
    void* args[] = {&probs,      &renormed_prob,     &top_k_arr,  &top_k_val,     &vocab_size,
                    &batch_size, &row_states_buffer, &chunk_size, &ctas_per_group, &k_safe, &visits, &track};
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
  });

  return cudaSuccess;
}
}}
