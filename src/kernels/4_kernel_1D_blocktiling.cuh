#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm1DBlocktilingOriginal(int M, int N, int K, float alpha,
                                           const float *A, const float *B, float beta,
                                           float *C)
{
  // If we flip x and y here we get ~30% less performance for large matrices.
  // The current, 30% faster configuration ensures that blocks with sequential
  // blockIDs access columns of B sequentially, while sharing the same row of A.
  // The slower configuration would share columns of A, but access into B would
  // be non-sequential. So the faster configuration has better spatial locality
  // and hence a greater L2 hit rate.
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // each warp will calculate 32*TM elements, with 32 being the columnar dim.
  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  // allocate space for the current blocktile in SMEM
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // todo: adjust this to each thread to load multiple entries and
  // better exploit the cache sizes
  assert(BM * BK == blockDim.x);
  assert(BN * BK == blockDim.x);
  const uint innerColA = threadIdx.x % BK; // warp-level GMEM coalescing
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColB = threadIdx.x % BN; // warp-level GMEM coalescing
  const uint innerRowB = threadIdx.x / BN;

  // allocate thread-local cache for results in registerfile
  float threadResults[TM] = {0.0};

  // outer loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK)
  {
    // populate the SMEM caches
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    // advance blocktile
    A += BK;
    B += BK * N;

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx)
    {
      // we make the dotproduct loop the outside loop, which facilitates
      // reuse of the Bs entry, which we can cache in a tmp var.
      float tmpB = Bs[dotIdx * BN + threadCol];
      for (uint resIdx = 0; resIdx < TM; ++resIdx)
      {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdx = 0; resIdx < TM; ++resIdx)
  {
    C[(threadRow * TM + resIdx) * N + threadCol] =
        alpha * threadResults[resIdx] +
        beta * C[(threadRow * TM + resIdx) * N + threadCol];
  }
}

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm1DBlocktiling(
    const int M, const int N, const int K,
    const float alpha,
    const float *A,
    const float *B,
    const float beta,
    float *C)
{
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  const uint c_col = blockIdx.x;
  const uint c_row = blockIdx.y;
  const uint thread_col = threadIdx.x % BN;
  const uint thread_row = threadIdx.x / BN;

  A += c_row * BM * K;
  B += c_col * BN;
  C += c_row * BM * N + c_col * BN;

  assert(BM * BK == blockDim.x);
  assert(BN * BK == blockDim.x);
  const uint inner_col_a = threadIdx.x % BK;
  const uint inner_row_a = threadIdx.x / BK;
  const uint inner_col_b = threadIdx.x % BN;
  const uint inner_row_b = threadIdx.x / BN;

  float thread_results[TM] = {0.0};
  for (uint bk_idx = 0; bk_idx < K; bk_idx += BK)
  {
    As[inner_row_a * BK + inner_col_a] = A[inner_row_a * K + inner_col_a];
    Bs[inner_row_b * BN + inner_col_b] = B[inner_row_b * N + inner_col_b];
    __syncthreads();

    for (uint dot_idx = 0; dot_idx < BK; dot_idx++)
    {
      float tmpB = Bs[dot_idx * BN + thread_col];
      for (uint res_idx = 0; res_idx < TM; res_idx++)
        thread_results[res_idx] += As[(thread_row * TM + res_idx) * BK + dot_idx] * tmpB;
    }
    __syncthreads();

    A += BK;
    B += BK * N;
  }

  for (uint res_idx = 0; res_idx < TM; res_idx++)
  {
    C[(thread_row * TM + res_idx) * N + thread_col] =
        alpha * thread_results[res_idx] +
        beta * C[(thread_row * TM + res_idx) * N + thread_col];
  }
}
