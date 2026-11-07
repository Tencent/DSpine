/*
 * Copyright (c) 2026 by FlashInfer team.
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
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
// Source: https://github.com/flashinfer-ai/flashinfer/blob/v0.7.1/include/flashinfer/air_top_p.cuh
// AIR Top-P originates in NVIDIA TensorRT-LLM; see THIRD_PARTY_NOTICES.txt.
// Modified for DSpine: add per-row early returns and visit counters.

namespace flashinfer { namespace sampling { namespace air_top_p {
template <typename T, typename HisT_>
__global__ void ConditionalAirTopPRenormInitKernel(Counter<T>* counters, int len, T const* in,
                                        float* top_p_arr, float top_p_val, HisT_* histograms,
                                        IdxT* countHistograms, bool const* certified, int* visits, bool track) {
  if (certified[blockIdx.x]) return;
  if (track && threadIdx.x == 0) visits[blockIdx.x * 3] = 1;
  auto const batchIdx = blockIdx.x;
  Counter<T>* counter = counters + batchIdx;
  if (threadIdx.x == 0) {
    counter->in = in + batchIdx * len;
    counter->len = len;
    counter->oriLen = len;
    counter->previousLen = len;
    counter->p = top_p_arr ? top_p_arr[batchIdx] : top_p_val;
    counter->sum = 0;
    counter->kthValueBits = 0;
    counter->finishedBlockCnt = 0;
    counter->filterCnt = 0;
  }
  HisT_* hist = histograms + batchIdx * NUM_BUCKETS;
  IdxT* cntHist = countHistograms + batchIdx * NUM_BUCKETS;
  for (int i = threadIdx.x; i < NUM_BUCKETS; i += blockDim.x) {
    hist[i] = 0;
    cntHist[i] = 0;
  }
}

template <bool IsDeterministic, typename T>
__global__ void ConditionalAirTopPRenormRadixKernel(Counter<T>* counters, HisT<IsDeterministic, T>* histograms,
                                         IdxT* countHistograms, int const pass, T* buf1, T* buf2, bool const* certified, int* visits, bool track) {
  if (certified[blockIdx.y]) return;
  if (track && threadIdx.x == 0) atomicAdd(visits + blockIdx.y * 3 + 1, 1);
  using HisT_ = HisT<IsDeterministic, T>;
  int const batchId = blockIdx.y;
  auto counter = counters + batchId;
  AccT currentSum;
  IdxT previousLen, currentLen;
  if (pass == 0) {
    currentSum = 0;
    previousLen = counter->len;
    currentLen = counter->len;
  } else {
    currentSum = counter->sum;
    currentLen = counter->len;
    previousLen = counter->previousLen;
  }
  if (currentLen == 0) return;
  IdxT const bufLen = calcBufLen<T>(counter->oriLen);
  T const* inBuf = nullptr;
  T* outBuf = nullptr;
  setBufPointers(counter->in, buf1 + bufLen * batchId, buf2 + bufLen * batchId, pass, inBuf,
                 outBuf);
  if (pass == 0 || pass == 1 || previousLen > bufLen) {
    inBuf = counter->in;
    previousLen = counter->oriLen;
  }
  if (pass == 0 || currentLen > bufLen) {
    outBuf = nullptr;
  }
  auto histogram = histograms + batchId * NUM_BUCKETS;
  auto countHistogram = countHistograms + batchId * NUM_BUCKETS;
  __shared__ HisT_ histogramSmem[NUM_BUCKETS];
  __shared__ IdxT countHistogramSmem[NUM_BUCKETS];
  filterAndHistogram<IsDeterministic, T, HisT_>(inBuf, outBuf, previousLen, counter, histogram,
                                                countHistogram, histogramSmem, countHistogramSmem,
                                                pass);
  __syncthreads();
  __threadfence();
  bool isLastBlock = false;
  if (threadIdx.x == 0) {
    uint32_t finished = atomicInc(&counter->finishedBlockCnt, gridDim.x - 1);
    isLastBlock = (finished == (gridDim.x - 1));
  }
  if (__syncthreads_or(isLastBlock)) {
    AccT* histValueSmem = reinterpret_cast<AccT*>(histogramSmem);
    if constexpr (IsDeterministic) {
      for (int i = threadIdx.x; i < NUM_BUCKETS; i += blockDim.x) {
        uint64_t value = static_cast<uint64_t>(histogram[i]);
        IdxT count = countHistogram[i];
        if (count != 0) {
          uint32_t sb = calcStartBit<T>(pass);
          uint32_t bv = counter->kthValueBits;
          if (pass == 0) bv = i << sb;
          histValueSmem[i] = calcFloatValue(static_cast<uint32_t>(count),
                                            calcExponent(twiddleOut<T>(bv, false)), value);
        } else {
          histValueSmem[i] = 0.0f;
        }
      }
      __syncthreads();
    }
    constexpr int WARP_SIZE = 32;
    constexpr int WARP_COUNT = NUM_BUCKETS / WARP_SIZE;
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    AccT* histPtr = IsDeterministic ? histValueSmem : reinterpret_cast<AccT*>(histogram);
    __shared__ AccT warpSum[WARP_COUNT];
    __shared__ cuda::atomic<AccT, cuda::thread_scope_block> blockSum;
    if constexpr (BITS_PER_PASS != 11) {
      for (int i = threadIdx.x; i < NUM_BUCKETS; i += BLOCK_SIZE) warpSum[i] = 0;
      __syncthreads();
    }
    for (int i = threadIdx.x; i < NUM_BUCKETS; i += BLOCK_SIZE)
      reduce_store_async(warp, warpSum + i / WARP_SIZE, histPtr[i], cg::plus<float>{});
    __syncthreads();
    if (threadIdx.x < WARP_SIZE) {
      reduce_store_async(warp, blockSum, warpSum[threadIdx.x], cg::plus<float>{});
      if constexpr (BITS_PER_PASS == 11)
        reduce_update_async(warp, blockSum, warpSum[threadIdx.x + WARP_SIZE], cg::plus<float>{});
    }
    __syncthreads();
    if (pass == 0) currentSum = blockSum * counter->p;
    if (threadIdx.x == 0) {
      AccT prev = 0;
      int targetStep = 0;
      for (int i = 0; i < WARP_COUNT; i++) {
        if (warpSum[i]) {
          targetStep = i;
          if ((prev + warpSum[i]) >= currentSum) break;
          prev += warpSum[i];
        }
      }
      int targetIdx = 0;
      for (int i = targetStep * WARP_SIZE; i < NUM_BUCKETS; i++) {
        if (countHistogram[i]) {
          targetIdx = i;
          if ((prev + histPtr[i]) >= currentSum) break;
          prev += histPtr[i];
        }
      }
      counter->sum = currentSum - prev;
      counter->len = countHistogram[targetIdx];
      typename cub::Traits<T>::UnsignedBits bucket = targetIdx;
      counter->kthValueBits |= bucket << calcStartBit<T>(pass);
    }
    __syncthreads();
    if (pass != NUM_PASSES<T> - 1) {
      for (int i = threadIdx.x; i < NUM_BUCKETS; i += blockDim.x) {
        histogram[i] = 0;
        countHistogram[i] = 0;
      }
    }
    if (threadIdx.x == 0) {
      counter->previousLen = currentLen;
      counter->filterCnt = 0;
    }
  }
}
}}}
