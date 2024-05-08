#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

namespace mine
{
    // transeposed As. 18551.4 -> 17107.0
    template <const int BM, const int BN, const int BK, const int TM, const int TN>
    __global__ void sgemmVectorize2(
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

        A += blockIdx.y * BM * K;
        B += blockIdx.x * BN;
        C += blockIdx.y * BM * N + blockIdx.x * BN;

        // TM-TN box => 한 thread에서 처리
        // block안에서 몇 번째 TM-TN box인지
        const uint thread_row = threadIdx.x / (BN / TN);
        const uint thread_col = threadIdx.x % (BN / TN);

        // gmem IO 할 때의 row/col
        const uint inner_row_A = threadIdx.x / (BK / 4);
        const uint inner_col_A = threadIdx.x % (BK / 4);
        const uint inner_row_B = threadIdx.x / (BN / 4);
        const uint inner_col_B = threadIdx.x % (BN / 4);

        for (uint k_idx = 0; k_idx < K; k_idx += BK)
        {
            // float4 tmp = reinterpret_cast<float4 *>(&A[inner_row_A * K + inner_col_A * 4])[0];
            // As[(inner_col_A * 4 + 0) * BM + inner_row_A] = tmp.x;
            // As[(inner_col_A * 4 + 1) * BM + inner_row_A] = tmp.y;
            // As[(inner_col_A * 4 + 2) * BM + inner_row_A] = tmp.z;
            // As[(inner_col_A * 4 + 3) * BM + inner_row_A] = tmp.w;
            reinterpret_cast<float4 *>(&As[inner_row_A * BK + inner_col_A * 4])[0] =
                reinterpret_cast<float4 *>(&A[inner_row_A * K + inner_col_A * 4])[0];

            reinterpret_cast<float4 *>(&Bs[inner_row_B * BN + inner_col_B * 4])[0] =
                reinterpret_cast<float4 *>(&B[inner_row_B * N + inner_col_B * 4])[0];
            __syncthreads();

            for (uint bk_idx = 0; bk_idx < BK; bk_idx++)
            {
                for (uint i = 0; i < TM; i++)
                    // regM[i] = As[bk_idx * BM + thread_row * TM + i];
                    regM[i] = As[(thread_row * TM + i) * BK + bk_idx]; // 느려짐 18.6K -> 17.1K
                // HBM->SMEM에선 어차피 SMEM이 압도적으로 빠르므로 HBM만 coalescing 하면 됨.
                // 마찬가지로 SMEM -> REG에서 REG가 빠르므로 SMEM에서 coalescing이 발생해야 빠름.
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

    // increase BK. error
    template <const int BM, const int BN, const int BK, const int TM, const int TN, const int STRIDE_A, const int STRIDE_B>
    __global__ void sgemmVectorize3(
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

        A += blockIdx.y * BM * K;
        B += blockIdx.x * BN;
        C += blockIdx.y * BM * N + blockIdx.x * BN;

        // TM-TN box => 한 thread에서 처리
        // block안에서 몇 번째 TM-TN box인지
        const uint thread_row = threadIdx.x / (BN / TN);
        const uint thread_col = threadIdx.x % (BN / TN);

        // gmem IO 할 때의 row/col
        const uint inner_row_A = threadIdx.x / (BK / 4);
        const uint inner_col_A = threadIdx.x % (BK / 4);
        const uint inner_row_B = threadIdx.x / (BN / 4);
        const uint inner_col_B = threadIdx.x % (BN / 4);

        for (uint k_idx = 0; k_idx < K; k_idx += BK)
        {
#pragma unroll
            for (uint i = 0; i < BM; i += STRIDE_A)
            {
                float4 tmp = reinterpret_cast<float4 *>(&A[(inner_row_A + i) * K + inner_col_A * 4])[0];
                As[(inner_col_A * 4 + 0) * BM + inner_row_A + i] = tmp.x;
                As[(inner_col_A * 4 + 1) * BM + inner_row_A + i] = tmp.y;
                As[(inner_col_A * 4 + 2) * BM + inner_row_A + i] = tmp.z;
                As[(inner_col_A * 4 + 3) * BM + inner_row_A + i] = tmp.w;
            }
#pragma unroll
            for (uint i = 0; i < BK; i += STRIDE_B)
            {
                reinterpret_cast<float4 *>(&Bs[(inner_row_B + i) * BN + inner_col_B * 4])[0] =
                    reinterpret_cast<float4 *>(&B[(inner_row_B + i) * N + inner_col_B * 4])[0];
            }
            __syncthreads();

            for (uint bk_idx = 0; bk_idx < BK; bk_idx++)
            {
                // for (uint i = 0; i < TM; i++)
                //     regM[i] = As[bk_idx * BM + thread_row * TM + i];
                // for (uint i = 0; i < TN; i++)
                //     regN[i] = Bs[bk_idx * BN + thread_col * TN + i];
                for (uint i = 0; i < TM; i += 4)
                    reinterpret_cast<float4 *>(&regM[i])[0] =
                        reinterpret_cast<float4 *>(&As[bk_idx * BM + thread_row * TM + i])[0];
                for (uint i = 0; i < TN; i += 4)
                    reinterpret_cast<float4 *>(&regN[i])[0] =
                        reinterpret_cast<float4 *>(&Bs[bk_idx * BN + thread_col * TN + i])[0];

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

    // vectorize (SMEM -> REG). 18551.4 -> 19038.2
    template <const int BM, const int BN, const int BK, const int TM, const int TN>
    __global__ void sgemmVectorize4(
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

        A += blockIdx.y * BM * K;
        B += blockIdx.x * BN;
        C += blockIdx.y * BM * N + blockIdx.x * BN;

        // TM-TN box => 한 thread에서 처리
        // block안에서 몇 번째 TM-TN box인지
        const uint thread_row = threadIdx.x / (BN / TN);
        const uint thread_col = threadIdx.x % (BN / TN);

        // gmem IO 할 때의 row/col
        const uint inner_row_A = threadIdx.x / (BK / 4);
        const uint inner_col_A = (threadIdx.x % (BK / 4)) * 4;
        const uint inner_row_B = threadIdx.x / (BN / 4);
        const uint inner_col_B = (threadIdx.x % (BN / 4)) * 4;

        for (uint k_idx = 0; k_idx < K; k_idx += BK)
        {
            const float4 tmp = reinterpret_cast<const float4 *>(&A[inner_row_A * K + inner_col_A])[0];
            As[(inner_col_A + 0) * BM + inner_row_A] = tmp.x;
            As[(inner_col_A + 1) * BM + inner_row_A] = tmp.y;
            As[(inner_col_A + 2) * BM + inner_row_A] = tmp.z;
            As[(inner_col_A + 3) * BM + inner_row_A] = tmp.w;

            reinterpret_cast<float4 *>(&Bs[inner_row_B * BN + inner_col_B])[0] =
                reinterpret_cast<float4 *>(&B[inner_row_B * N + inner_col_B])[0];
            __syncthreads();

            for (uint bk_idx = 0; bk_idx < BK; bk_idx++)
            {
                for (uint i = 0; i < TM; i++)
                    regM[i] = As[bk_idx * BM + thread_row * TM + i];
                for (uint i = 0; i < TN; i++)
                    regN[i] = Bs[bk_idx * BN + thread_col * TN + i];
                // for (uint i = 0; i < TM; i += 4)
                //     reinterpret_cast<float4 *>(&regM[i])[0] =
                //         reinterpret_cast<float4 *>(&As[bk_idx * BM + thread_row * TM + i])[0];
                // for (uint i = 0; i < TN; i += 4)
                //     reinterpret_cast<float4 *>(&regN[i])[0] =
                //         reinterpret_cast<float4 *>(&Bs[bk_idx * BN + thread_col * TN + i])[0];
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
}
