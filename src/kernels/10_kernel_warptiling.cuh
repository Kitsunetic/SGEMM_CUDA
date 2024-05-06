#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
const int WARPSIZE = 32; // warpSize is not constexpr

namespace wt_original
{
  template <const int BM, const int BN, const int BK, const int rowStrideA,
            const int rowStrideB>
  __device__ void loadFromGmemOriginal(int N, int K, const float *A, const float *B,
                                       float *As, float *Bs, int innerRowA, int innerColA,
                                       int innerRowB, int innerColB)
  {
    for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA)
    {
      const float4 tmp = reinterpret_cast<const float4 *>(
          &A[(innerRowA + offset) * K + innerColA * 4])[0];
      // float4 tmp;
      // asm("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];"
      //     : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
      //     : "l"(&A[(innerRowA + offset) * K + innerColA * 4]));
      As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
      As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
      As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
      As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
    }

    for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB)
    {
      reinterpret_cast<float4 *>(
          &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
          reinterpret_cast<const float4 *>(
              &B[(innerRowB + offset) * N + innerColB * 4])[0];
      // asm("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
      //     : "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 0]),
      //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 1]),
      //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 2]),
      //       "=f"(Bs[(innerRowB + offset) * BN + innerColB * 4 + 3])
      //     : "l"(&B[(innerRowB + offset) * N + innerColB * 4]));
    }
  }

  template <const int BM, const int BN, const int BK, const int WM, const int WN,
            const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
            const int TM, const int TN>
  __device__ void
  processFromSmemOriginal(float *regM, float *regN, float *threadResults, const float *As,
                          const float *Bs, const uint warpRow, const uint warpCol,
                          const uint threadRowInWarp, const uint threadColInWarp)
  {
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx)
    {
      // populate registers for whole warptile
      for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
      {
        for (uint i = 0; i < TM; ++i)
        {
          regM[wSubRowIdx * TM + i] =
              As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
                 threadRowInWarp * TM + i];
        }
      }
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
      {
        for (uint i = 0; i < TN; ++i)
        {
          regN[wSubColIdx * TN + i] =
              Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
                 threadColInWarp * TN + i];
        }
      }

      // execute warptile matmul
      for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
      {
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
        {
          // calculate per-thread results
          for (uint resIdxM = 0; resIdxM < TM; ++resIdxM)
          {
            for (uint resIdxN = 0; resIdxN < TN; ++resIdxN)
            {
              threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                            (wSubColIdx * TN) + resIdxN] +=
                  regM[wSubRowIdx * TM + resIdxM] *
                  regN[wSubColIdx * TN + resIdxN];
            }
          }
        }
      }
    }
  }

} // namespace wt

/*
 * @tparam BM The threadblock size for M dimension SMEM caching.
 * @tparam BN The threadblock size for N dimension SMEM caching.
 * @tparam BK The threadblock size for K dimension SMEM caching.
 * @tparam WM M dim of continuous tile computed by each warp
 * @tparam WN N dim of continuous tile computed by each warp
 * @tparam WMITER The number of subwarp tiling steps in M dimension.
 * @tparam WNITER The number of subwarp tiling steps in N dimension.
 * @tparam TM The per-thread tile size for M dimension.
 * @tparam TN The per-thread tile size for N dimension.
 */
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    sgemmWarptilingOriginal(int M, int N, int K, float alpha, float *A, float *B,
                            float beta, float *C)
{
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // Placement of the warp in the threadblock tile
  const uint warpIdx = threadIdx.x / WARPSIZE; // the warp this thread is in
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);

  // size of the warp subtile
  constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER; // 64/2=32
  constexpr uint WSUBN = WN / WNITER; // 32/2=16

  // Placement of the thread in the warp subtile
  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;         // [0, 31]
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN); // i%(16/4)
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN); // i/4

  // allocate space for the current blocktile in SMEM
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  // Move C_ptr to warp's output tile
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  // calculating the indices that this thread will load into SMEM
  // we'll load 128bit / 32bit = 4 elements per thread at each step
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

  // allocate thread-local cache for results in registerfile
  float threadResults[WMITER * TM * WNITER * TN] = {0.0};
  // we cache into registers on the warptile level
  float regM[WMITER * TM] = {0.0};
  float regN[WNITER * TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK)
  {
    wt_original::loadFromGmemOriginal<BM, BN, BK, rowStrideA, rowStrideB>(
        N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
    __syncthreads();

    wt_original::processFromSmemOriginal<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
        regM, regN, threadResults, As, Bs, warpRow, warpCol, threadRowInWarp, threadColInWarp);
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down
    __syncthreads();
  }

  // write out the results
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
  {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
    {
      // move C pointer to current warp subtile
      float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1)
      {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4)
        {
          // load C vector into registers
          float4 tmp = reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0];
          // perform GEMM update in reg
          const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
          tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
          tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
          tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
          tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
          // write back
          reinterpret_cast<float4 *>(
              &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                         threadColInWarp * TN + resIdxN])[0] = tmp;
        }
      }
    }
  }
}

namespace wt
{
  template <const int BM, const int BN, const int BK, const int row_stride_A, const int row_stride_B>
  __device__ void loadFromGmem(
      const int N, const int K,
      const float *A,
      const float *B,
      float *As,
      float *Bs,
      uint inner_row_A,
      uint inner_col_A,
      uint inner_row_B,
      uint inner_col_B)
  {
    for (uint offset = 0; offset + row_stride_A <= BM; offset += row_stride_A)
    {
      const float4 tmp = reinterpret_cast<const float4 *>(&A[(inner_row_A + offset) * K + inner_col_A * 4])[0];
      As[(inner_col_A * 4 + 0) * BM + inner_row_A + offset] = tmp.x;
      As[(inner_col_A * 4 + 1) * BM + inner_row_A + offset] = tmp.y;
      As[(inner_col_A * 4 + 2) * BM + inner_row_A + offset] = tmp.z;
      As[(inner_col_A * 4 + 3) * BM + inner_row_A + offset] = tmp.w;
    }

    for (uint offset = 0; offset + row_stride_B < BK; offset += row_stride_B)
    {
      reinterpret_cast<float4 *>(&Bs[(inner_row_B + offset) * BN + inner_col_B * 4])[0] =
          reinterpret_cast<const float4 *>(&B[(inner_row_B + offset) * N + inner_col_B * 4])[0];
    }
  }

  template <const int BM, const int BN, const int BK,
            const int WM, const int WN, const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
            const int TM, const int TN>
  __device__ void processFromSmem(
      float *regM, float *regN, float *thread_results,
      const float *As, const float *Bs,
      const uint warp_row, const uint warp_col,
      const uint thread_row_in_warp, const uint thread_col_in_warp)
  {
    for (uint bk_idx = 0; bk_idx < BK; bk_idx++)
    {
      for (uint w_sub_row_idx = 0; w_sub_row_idx < WMITER; w_sub_row_idx++)
        for (uint i = 0; i < TM; i++)
          regM[w_sub_row_idx * TM + i] =
              As[bk_idx * BM + warp_row * WM + w_sub_row_idx * WSUBM + thread_row_in_warp * TM + i];
      for (uint w_sub_col_idx = 0; w_sub_col_idx < WNITER; w_sub_col_idx++)
        for (uint i = 0; i < TN; i++)
          regN[w_sub_col_idx * TN + i] =
              Bs[bk_idx * BN + warp_col * WN + w_sub_col_idx * WSUBN + thread_col_in_warp * TN + i];

      for (uint w_sub_row_idx = 0; w_sub_row_idx < WMITER; w_sub_row_idx++)
        for (uint w_sub_col_idx = 0; w_sub_col_idx < WNITER; w_sub_col_idx++)
          for (uint res_idx_m = 0; res_idx_m < TM; res_idx_m++)
            for (uint res_idx_n = 0; res_idx_n < TN; res_idx_n++)
              thread_results[(w_sub_row_idx * TM + res_idx_m) * (WNITER * TN) + (w_sub_col_idx * TN) + res_idx_n] +=
                  regM[w_sub_row_idx * TM + res_idx_m] * regN[w_sub_col_idx * TN + res_idx_n];
    }
  }

}

template <const int BM, const int BN, const int BK,
          const int WM, const int WN, const int WNITER,
          const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS) sgemmWarptiling(
    const int M, const int N, const int K,
    const float alpha,
    const float *A,
    const float *B,
    const float beta,
    float *C)
{
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  const uint c_row = blockIdx.y;
  const uint c_col = blockIdx.x;

  const uint warp_idx = threadIdx.x / WARPSIZE; // block 내에서 몇 번째 warp인지
  const uint warp_col = warp_idx % (BN / WN);   // thread를 warp단위로 끊었을 때의 row/col
  const uint warp_row = warp_idx / (BN / WN);

  constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER;
  constexpr uint WSUBN = WN / WNITER;

  const uint thread_idx_in_warp = threadIdx.x % WARPSIZE;
  const uint thread_col_in_warp = thread_idx_in_warp % (WSUBN / TN);
  const uint thread_row_in_warp = thread_idx_in_warp / (WSUBN / TN);

  A += c_row * BM * K;
  B += c_col * BN;
  C += (c_row * BM + warp_row * WM) * N + c_col * BN + warp_col * WN;

  const uint inner_row_A = threadIdx.x / (BK / 4);
  const uint inner_col_A = threadIdx.x % (BK / 4);
  constexpr uint row_stride_A = (NUM_THREADS * 4) / BK;
  const uint inner_row_B = threadIdx.x / (BN / 4);
  const uint inner_col_B = threadIdx.x % (BN / 4);
  constexpr uint row_stride_B = NUM_THREADS / (BN / 4);

  float thread_results[WMITER * TM * WNITER * TN] = {0.0};
  float regM[WMITER * TM];
  float regN[WNITER * TN];

  for (uint k_idx = 0; k_idx < K; k_idx += BK)
  {
    wt::loadFromGmem<BM, BN, BK, row_stride_A, row_stride_B>(
        N, K, A, B, As, Bs, inner_row_A, inner_col_A, inner_row_B, inner_col_B);
    __syncthreads();

    wt::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>(
        regM, regN, thread_results, As, Bs, warp_row, warp_col, thread_row_in_warp, thread_col_in_warp);
    __syncthreads();

    A += BK;
    B += BK * N;
  }

  for (uint w_sub_row_idx = 0; w_sub_row_idx < WMITER; w_sub_row_idx++)
    for (uint w_sub_col_idx = 0; w_sub_col_idx < WNITER; w_sub_col_idx++)
    {
      float *C_interim = C + (w_sub_row_idx * WSUBM) * N + w_sub_col_idx * WSUBN;
      for (uint res_idx_M = 0; res_idx_M < TM; res_idx_M++)
        for (uint res_idx_N = 0; res_idx_N < TN; res_idx_N += 4)
        {
          float4 tmp = reinterpret_cast<float4 *>(
              &C_interim[(thread_row_in_warp * TM + res_idx_M) * N + thread_col_in_warp * TN + res_idx_N])[0];
          const int i = (w_sub_col_idx * TM + res_idx_M) * (WNITER * TN) + w_sub_col_idx * TN + res_idx_N;
          tmp.x = alpha * thread_results[i + 0] + beta * tmp.x;
          tmp.y = alpha * thread_results[i + 1] + beta * tmp.y;
          tmp.z = alpha * thread_results[i + 2] + beta * tmp.z;
          tmp.w = alpha * thread_results[i + 3] + beta * tmp.w;
          reinterpret_cast<float4 *>(
              &C_interim[(thread_row_in_warp * TM + res_idx_M) * N + thread_col_in_warp * TN + res_idx_N])[0] = tmp;
        }
    }
}
