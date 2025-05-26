#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include "kernels.cuh"

// V6: cuBLAS implementation - reference implementation
void runV6CuBLAS(const float* d_A, const float* d_B, float* d_C, int N, double& time_ms, double& gflops) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const float alpha = 1.0f, beta = 0.0f;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // cuBLAS uses column-major order, so we need to transpose the operation
    // C = A * B becomes C^T = B^T * A^T in column-major
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, 
                &alpha, d_B, N, d_A, N, &beta, d_C, N);
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    gflops = 2.0 * N * N * N / (time_ms / 1000.0) / 1e9;
    
    cublasDestroy(handle);
}
