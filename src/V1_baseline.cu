#include <cuda_runtime.h>
#include "kernels.cuh"

// V1: Baseline kernel - each thread computes one element
__global__ void V1_baselineKernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Wrapper function for V1
void runV1Baseline(const float* d_A, const float* d_B, float* d_C, int N, int blockSize) {
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 blocksPerGrid((N + blockSize - 1) / blockSize, (N + blockSize - 1) / blockSize);
    V1_baselineKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
}
