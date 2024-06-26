#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*

Matrix sizes:
MxK * KxN = MxN

*/

// __global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
//                             const float *B, float beta, float *C)
// {
//   const uint x = blockIdx.x * blockDim.x + threadIdx.x;
//   const uint y = blockIdx.y * blockDim.y + threadIdx.y;

//   // if statement is necessary to make things work under tile quantization
//   if (x < M && y < N)
//   {
//     float tmp = 0.0;
//     for (int i = 0; i < K; ++i)
//     {
//       tmp += A[x * K + i] * B[i * N + y];
//     }
//     // C = α*(A@B)+β*C
//     C[x * N + y] = alpha * tmp + beta * C[x * N + y];
//   }
// }

__global__ void sgemm_naive_original(
    const int M, const int N, const int K,
    const float alpha,
    const float *A,
    const float *B,
    const float beta,
    float *C)
{
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= M || y >= N)
    return;

  float tmp = 0.0;
  for (int i = 0; i < K; i++)
    tmp += A[x * K + i] * B[i * N + y];
  C[x * N + y] = alpha * tmp + beta * C[x * N + y];
}

__global__ void sgemm_naive(
    const int M, const int N, const int K,
    const float alpha,
    const float *A,
    const float *B,
    const float beta,
    float *C)
{
  const int y = blockIdx.x * blockDim.x + threadIdx.x;
  const int x = blockIdx.y * blockDim.y + threadIdx.y;
  // const uint y = blockIdx.x * 32 + threadIdx.x;
  // const uint x = blockIdx.y * 32 + threadIdx.y;
  if (x >= M || y >= N)
    return;

  float tmp = 0.0;
  for (int i = 0; i < K; i++)
    tmp += A[x * K + i] * B[i * N + y];
  C[x * N + y] = alpha * tmp + beta * C[x * N + y];
}
