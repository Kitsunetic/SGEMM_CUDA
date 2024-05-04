#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C)
{
  // the output block that we want to compute in this threadblock
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  // allocate buffer for current block in fast shared mem
  // shared mem is shared between all threads in a block
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  // the inner row & col that we're accessing in this thread
  const uint threadCol = threadIdx.x % BLOCKSIZE;
  const uint threadRow = threadIdx.x / BLOCKSIZE;

  // advance pointers to the starting positions
  A += cRow * BLOCKSIZE * K;                    // row=cRow, col=0
  B += cCol * BLOCKSIZE;                        // row=0, col=cCol
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

  float tmp = 0.0;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE)
  {
    // Have each thread load one of the elements in A & B
    // Make the threadCol (=threadIdx.x) the consecutive index
    // to allow global memory access coalescing
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

    // block threads in this block until cache is fully populated
    __syncthreads();
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    // execute the dotproduct on the currently cached block
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx)
    {
      tmp += As[threadRow * BLOCKSIZE + dotIdx] *
             Bs[dotIdx * BLOCKSIZE + threadCol];
    }
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done
    __syncthreads();
  }
  C[threadRow * N + threadCol] =
      alpha * tmp + beta * C[threadRow * N + threadCol];
}

template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_block(
    int M, int N, int K,
    float alpha,
    const float *A,
    const float *B,
    float beta,
    float *__restrict__ C)
{
  __shared__ float As[BLOCKSIZE * BLOCKSIZE]; // static shared memory allocation
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  const uint thread_col = threadIdx.x % BLOCKSIZE;
  const uint thread_row = threadIdx.x / BLOCKSIZE;

  A += blockIdx.x * BLOCKSIZE * K;                          // row of A
  B += blockIdx.x * BLOCKSIZE;                              // column of B
  C += blockIdx.x * BLOCKSIZE * N + blockIdx.y * BLOCKSIZE; // row + column of C

  float tmp = 0.0;
  for (uint bk_idx = 0; bk_idx < K; bk_idx += BLOCKSIZE)
  {
    // It can be faster than previous due to coalescing of both A and B and reusing.
    As[thread_row * BLOCKSIZE + thread_col] = A[thread_row * K + thread_col];
    Bs[thread_row * BLOCKSIZE + thread_col] = B[thread_row * N + thread_col];
    __syncthreads();

    for (int dot_idx = 0; dot_idx < BLOCKSIZE; dot_idx++)
      tmp += As[thread_row * BLOCKSIZE + dot_idx] * Bs[dot_idx * BLOCKSIZE + thread_col];
    __syncthreads();

    A += BLOCKSIZE;
    B += BLOCKSIZE * N;
  }

  C[thread_row * N + thread_col] = alpha * tmp + beta * C[thread_row * N + thread_col];
}
