#include <cuda_runtime.h>
#include "kernels.cuh"

#define COARSE_FACTOR 4

// V4: Thread coarsening kernel - each thread computes multiple elements
__global__ void V4_threadCoarseningKernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col_start = (blockIdx.x * blockDim.x + threadIdx.x) * COARSE_FACTOR;

    if (row < N) {
        for (int c = 0; c < COARSE_FACTOR; ++c) {
            int col = col_start + c;
            if (col < N) {
                float sum = 0.0f;
                for (int k = 0; k < N; ++k) {
                    sum += A[row * N + k] * B[k * N + col];
                }
                C[row * N + col] = sum;
            }
        }
    }
}

// Wrapper function for V4
void runV4ThreadCoarsening(const float* d_A, const float* d_B, float* d_C, int N, int blockSize) {
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 blocksPerGrid((N + blockSize * COARSE_FACTOR - 1) / (blockSize * COARSE_FACTOR), 
                       (N + blockSize - 1) / blockSize);
    V4_threadCoarseningKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
}