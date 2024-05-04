#pragma once

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// template <const uint BLOCKSIZE>
// __global__ void sgemm_global_mem_coalesce(int M, int N, int K, float alpha,
//                                           const float *A, const float *B,
//                                           float beta, float *C)
// {
//   const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
//   const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

//   // if statement is necessary to make things work under tile quantization
//   if (cRow < M && cCol < N)
//   {
//     float tmp = 0.0;
//     for (int i = 0; i < K; ++i)
//     {
//       tmp += A[cRow * K + i] * B[i * N + cCol];
//     }
//     C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
//   }
// }

template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(int M, int N, int K, float alpha,
                                          const float *__restrict__ A, const float *__restrict__ B,
                                          float beta, float *C)
{
  const int c_row = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int c_col = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
  if (c_row >= M || c_col >= N)
    return;

  float tmp = 0.0;
  for (int i = 0; i < K; i++)
  {
    tmp += A[c_row * K + i] * B[i * N + c_col];
  }
  C[c_row * N + c_col] = alpha * tmp + beta * C[c_row * N + c_col];
}
