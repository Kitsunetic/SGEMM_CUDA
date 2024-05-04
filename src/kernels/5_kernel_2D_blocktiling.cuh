#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm2DBlocktilingOriginal(int M, int N, int K, float alpha, const float *A,
                               const float *B, float beta, float *C)
{
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint totalResultsBlocktile = BM * BN;
  // A thread is responsible for calculating TM*TN elements in the blocktile
  const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

  // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
  assert(numThreadsBlocktile == blockDim.x);

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // allocate space for the current blocktile in smem
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // calculating the indices that this thread will load into SMEM
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  // calculates the number of rows of As that are being loaded in a single step
  // by a single block
  const uint strideA = numThreadsBlocktile / BK;
  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;
  // for both As and Bs we want each load to span the full column-width, for
  // better GMEM coalescing (as opposed to spanning full row-width and iterating
  // across columns)
  const uint strideB = numThreadsBlocktile / BN;

  // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN] = {0.0};
  // register caches for As and Bs
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK)
  {
    // populate the SMEM caches
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA)
    {
      As[(innerRowA + loadOffset) * BK + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB)
    {
      Bs[(innerRowB + loadOffset) * BN + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx)
    {
      // block into registers
      for (uint i = 0; i < TM; ++i)
      {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }
      for (uint i = 0; i < TN; ++i)
      {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM)
      {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN)
        {
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM)
  {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN)
    {
      C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
          alpha * threadResults[resIdxM * TN + resIdxN] +
          beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
    }
  }
}

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm2DBlocktiling(
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

  const uint total_results_blocktile = BM * BN;                           // 한 block에서 계산되는 C의 개수
  const uint num_threads_blocktile = total_results_blocktile / (TM * TN); // thread의 개수 -> 왜 그냥 blockDim.x를 안 쓰지?
  assert(num_threads_blocktile == blockDim.x);

  const uint thread_col = threadIdx.x % (BN / TN);
  const uint thread_row = threadIdx.x / (BN / TN);

  A += c_row * BM * K;
  B += c_col * BN;
  C += c_row * BM * N + c_col * BN;

  const uint inner_row_A = threadIdx.x / BK;
  const uint inner_col_A = threadIdx.x % BK;
  const uint stride_A = num_threads_blocktile / BK;
  const uint inner_row_B = threadIdx.x / BN;
  const uint inner_col_B = threadIdx.x % BN;
  const uint stride_B = num_threads_blocktile / BN;

  float thread_results[TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  for (uint bk_idx = 0; bk_idx < K; bk_idx += BK)
  {
    for (uint i = 0; i < BM; i += stride_A)
      As[(inner_row_A + i) * BK + inner_col_A] = A[(inner_row_A + i) * K + inner_col_A];
    for (uint i = 0; i < BK; i += stride_B)
      Bs[(inner_row_B + i) * BN + inner_col_B] = B[(inner_row_B + i) * N + inner_col_B];
    __syncthreads();

    for (uint dot_idx = 0; dot_idx < BK; dot_idx++)
    {
      for (uint i = 0; i < TM; i++)
        regM[i] = As[(thread_row * TM + i) * BK + dot_idx];
      for (uint i = 0; i < TM; i++)
        regN[i] = Bs[dot_idx * BN + thread_col * TN + i];
      for (uint res_idx_m = 0; res_idx_m < TM; res_idx_m++)
        for (uint res_idx_n = 0; res_idx_n < TN; res_idx_n++)
        {
          thread_results[res_idx_m * TN + res_idx_n] += regM[res_idx_m] * regN[res_idx_n];
        }
    }
    __syncthreads();

    A += BK;
    B += BK * N;
  }

  for (uint res_idx_m = 0; res_idx_m < TM; res_idx_m++)
    for (uint res_idx_n = 0; res_idx_n < TN; res_idx_n++)
    {
      C[(thread_row * TM + res_idx_m) * N + thread_col * TN + res_idx_n] =
          alpha * thread_results[res_idx_m * TN + res_idx_n] +
          beta * C[(thread_row * TM + res_idx_m) * N + thread_col * TN + res_idx_n];
    }
}
