#include <cuda_runtime.h>
#include "kernels.cuh"
#include <cstdio>

// #define TILE_SIZE 16
#define REG_TILE_SIZE 4

template <int TILE_SIZE>
// V5: Privatization kernel - register tiling with thread coarsening
__global__ void V5_privatizationKernel(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float results[REG_TILE_SIZE] = {0.0f};

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load data into shared memory
        if (row < N && t * TILE_SIZE + threadIdx.x < N) {
            As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        for (int r = 0; r < REG_TILE_SIZE; ++r) {
            int b_row = t * TILE_SIZE + threadIdx.y;
            int b_col = col + r * TILE_SIZE;
            if (b_row < N && b_col < N) {
                Bs[threadIdx.y][threadIdx.x] = B[b_row * N + b_col];
            } else {
                Bs[threadIdx.y][threadIdx.x] = 0.0f;
            }

            __syncthreads();

            for (int k = 0; k < TILE_SIZE; ++k) {
                results[r] += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            }

            __syncthreads();
        }
    }

    // Write results
    for (int r = 0; r < REG_TILE_SIZE; ++r) {
        int out_col = col + r * TILE_SIZE;
        if (row < N && out_col < N) {
            C[row * N + out_col] = results[r];
        }
    }
}

void runV5Privatization(const float* d_A, const float* d_B, float* d_C, int N, int blockSize) {
    // Always use TILE_SIZE for this kernel, ignore the blockSize parameter
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 blocksPerGrid((N + blockSize - 1) / blockSize, (N + blockSize - 1) / blockSize);
     if (blockSize == 8) {
        V5_privatizationKernel<8><<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    } else if (blockSize == 16) {
        V5_privatizationKernel<16><<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    } else if (blockSize == 32) {
        V5_privatizationKernel<32><<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    } else {
        printf("Error: Unsupported block size %d. Supported sizes: 8, 16, 32\n", blockSize);
        return;
    }

    cudaDeviceSynchronize();
}