#include <cuda_runtime.h>
#include "kernels.cuh"
#include <cstdio>

template <int TILE_SIZE>
__global__ void V6FinalKernel(const float* __restrict__ A, 
                                    const float* __restrict__ B, 
                                    float* __restrict__ C, 
                                    int N) {
    // Use exactly the same shared memory pattern as reference
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE + 1];  // +1 padding
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE + 1];  // +1 padding
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;
        // Add prefetching variables
    float next_A = 0.0f, next_B = 0.0f;
    // Ensure perfect memory coalescing like reference implementation
    for (int k = 0; k < N; k += TILE_SIZE) {

            // Prefetch next iteration data while current computation happens
        if (k + TILE_SIZE < N) {
            if (row < N && (k + TILE_SIZE + tx) < N) {
                next_A = A[row * N + k + TILE_SIZE + tx];
            }
            if ((k + TILE_SIZE + ty) < N && col < N) {
                next_B = B[(k + TILE_SIZE + ty) * N + col];
            }
        }

        // Load tiles with optimal access patterns
        if (row < N && (k + tx) < N) {
            tile_A[ty][tx] = A[row * N + k + tx];
        } else {
            tile_A[ty][tx] = 0.0f;
        }
        
        if ((k + ty) < N && col < N) {
            tile_B[ty][tx] = B[(k + ty) * N + col];
        } else {
            tile_B[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Unrolled inner loop for maximum throughput
        float sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f, sum4 = 0.0f;

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i += 4) {
            float a1 = tile_A[ty][i];
            float a2 = tile_A[ty][i + 1];
            float a3 = tile_A[ty][i + 2];
            float a4 = tile_A[ty][i + 3];
            
            float b1 = tile_B[i][tx];
            float b2 = tile_B[i + 1][tx];
            float b3 = tile_B[i + 2][tx];
            float b4 = tile_B[i + 3][tx];
            
            sum1 += a1 * b1;
            sum2 += a2 * b2;
            sum3 += a3 * b3;
            sum4 += a4 * b4;
        }
        sum += sum1 + sum2 + sum3 + sum4;
        
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

void runV6Final(const float* d_A, const float* d_B, float* d_C, int N, int blockSize) {
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 blocksPerGrid((N + blockSize - 1) / blockSize, (N + blockSize - 1) / blockSize);
    
    V6FinalKernel<32><<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
}