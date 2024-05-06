#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemmVectorizeOriginal(int M, int N, int K, float alpha, float *A,
                                       float *B, float beta, float *C)
{
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

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
  // we'll load 128bit / 32bit = 4 elements per thread at each step
  // TODO: BK=8 밖에 안 되고, BK/4=2라서 coalesce가 잘 안 이뤄짐
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);

  // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK)
  {
    // populate the SMEM caches
    // transpose A while loading it
    float4 tmp =
        reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

    reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
        reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];
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
        regM[i] = As[dotIdx * BM + threadRow * TM + i];
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
  for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1)
  {
    for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4)
    {
      // load C vector into registers
      float4 tmp = reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
      // perform GEMM update in reg
      tmp.x = alpha * threadResults[resIdxM * TN + resIdxN] + beta * tmp.x;
      tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
      tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
      tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;
      // write back
      reinterpret_cast<float4 *>(
          &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] =
          tmp;
    }
  }
}

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemmVectorize(
    const int M, const int N, const int K,
    const float alpha,
    float *A,
    float *B,
    const float beta,
    float *C)
{
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];
  float regM[TM];
  float regN[TN];
  float thread_results[TM * TN] = {0.0};

  const uint c_row = blockIdx.y;
  const uint c_col = blockIdx.x;

  A += c_row * BM * K;
  B += c_col * BN;
  C += c_row * BM * N + c_col * BN;

  const uint thread_row = threadIdx.x / (BN / TN);
  const uint thread_col = threadIdx.x % (BN / TN);

  const uint inner_row_A = threadIdx.x / (BK / 4);
  const uint inner_col_A = threadIdx.x % (BK / 4);
  const uint inner_row_B = threadIdx.x / (BN / 4);
  const uint inner_col_B = threadIdx.x % (BN / 4);

  for (uint k_idx = 0; k_idx < K; k_idx += BK)
  {
    float4 tmp = reinterpret_cast<float4 *>(&A[inner_row_A * K + inner_col_A * 4])[0];
    // 여기서 coalescing이 잘 안되는데, BK가 더 크고, 대신 BM & BN을 좀 줄여야 할 듯
    // BK=128, BM=32, BN=32?
    As[(inner_col_A * 4 + 0) * BM + inner_row_A] = tmp.x;
    As[(inner_col_A * 4 + 1) * BM + inner_row_A] = tmp.y;
    As[(inner_col_A * 4 + 2) * BM + inner_row_A] = tmp.z;
    As[(inner_col_A * 4 + 3) * BM + inner_row_A] = tmp.w;
    // 왜 row가 앞에 안오는거지?

    reinterpret_cast<float4 *>(&Bs[inner_row_B * BN + inner_col_B * 4])[0] =
        reinterpret_cast<float4 *>(&B[inner_row_B * N + inner_col_B * 4])[0];
    __syncthreads();

    for (uint bk_idx = 0; bk_idx < BK; bk_idx++)
    {
      for (uint i = 0; i < TM; i++)
        regM[i] = As[bk_idx * BM + thread_row * TM + i];
      for (uint i = 0; i < TN; i++)
        regN[i] = Bs[bk_idx * BN + thread_col * TN + i];
      for (uint res_idx_m = 0; res_idx_m < TM; res_idx_m++)
        for (uint res_idx_n = 0; res_idx_n < TN; res_idx_n++)
          thread_results[res_idx_m * TN + res_idx_n] += regM[res_idx_m] * regN[res_idx_n];
    }
    __syncthreads();

    A += BK;
    B += BK * N;
  }

  for (uint res_idx_m = 0; res_idx_m < TM; res_idx_m++)
    for (uint res_idx_n = 0; res_idx_n < TN; res_idx_n += 4)
    {
      float4 tmp = reinterpret_cast<float4 *>(&C[(thread_row * TM + res_idx_m) * N + thread_col * TN + res_idx_n])[0];
      tmp.x = alpha * thread_results[res_idx_m * TN + res_idx_n + 0] + beta * tmp.x;
      tmp.y = alpha * thread_results[res_idx_m * TN + res_idx_n + 1] + beta * tmp.y;
      tmp.z = alpha * thread_results[res_idx_m * TN + res_idx_n + 2] + beta * tmp.z;
      tmp.w = alpha * thread_results[res_idx_m * TN + res_idx_n + 3] + beta * tmp.w;

      reinterpret_cast<float4 *>(&C[(thread_row * TM + res_idx_m) * N + thread_col * TN + res_idx_n])[0] = tmp;
    }
}
