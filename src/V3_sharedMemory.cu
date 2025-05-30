#include <cuda_runtime.h>
#include "kernels.cuh"
#include <cstdio>

// #define TILE_SIZE 32
template <int TILE_SIZE>
// V3: Shared memory kernel for memory coalescing optimization
__global__ void V3_sharedMemoryKernel(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        if (row < N && t * TILE_SIZE + threadIdx.x < N) {
            As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && t * TILE_SIZE + threadIdx.y < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial sum using shared memory
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Wrapper function for V3
void runV3SharedMemory(const float* d_A, const float* d_B, float* d_C, int N, int blockSize) {

    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 blocksPerGrid((N + blockSize-1) / blockSize, (N + blockSize-1) / blockSize);

    if (blockSize == 8) {
        V3_sharedMemoryKernel<8><<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    } else if (blockSize == 16) {
        V3_sharedMemoryKernel<16><<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    } else if (blockSize == 32) {
        V3_sharedMemoryKernel<32><<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    } else {
        printf("Error: Unsupported block size %d. Supported sizes: 8, 16, 32\n", blockSize);
        return;
    }

    cudaDeviceSynchronize();
}