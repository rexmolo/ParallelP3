#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include "kernels.cuh"

void runKernelComparison(int N, int blockSize) {
    size_t bytes = N * N * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f + (float)(rand() % 100) / 100.0f;
        h_B[i] = 2.0f + (float)(rand() % 100) / 100.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    struct KernelTest {
        const char* name;
        void (*func)(const float*, const float*, float*, int, int);
    };

    std::vector<KernelTest> kernels = {
        {"V1_Base", runV1Baseline},
        {"V2_Unroll", runV2LoopUnroll},
        {"V3_Shared", runV3SharedMemory},
        {"V4_Coarse", runV4ThreadCoarsening},
        {"V5_Privat", runV5Privatization}
    };

    for (auto& test : kernels) {
        // Warmup
        test.func(d_A, d_B, d_C, N, blockSize);

        auto start = std::chrono::high_resolution_clock::now();
        test.func(d_A, d_B, d_C, N, blockSize);
        auto end = std::chrono::high_resolution_clock::now();

        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = 2.0 * N * N * N / (time_ms / 1000.0) / 1e9;
        
        printPerformanceRow(test.name, N, blockSize, time_ms, gflops);
    }

    // Test cuBLAS separately
    double cublas_time, cublas_gflops;
    runV6CuBLAS(d_A, d_B, d_C, N, cublas_time, cublas_gflops);
    printPerformanceRow("V6_cuBLAS", N, blockSize, cublas_time, cublas_gflops);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        return -1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("CUDA Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    printPerformanceHeader();

    std::vector<int> matrixSizes = {512, 1024, 2048};
    std::vector<int> blockSizes = {16, 32};

    for (int N : matrixSizes) {
        for (int blockSize : blockSizes) {
            if (blockSize * blockSize <= prop.maxThreadsPerBlock) {
                runKernelComparison(N, blockSize);
            }
        }
        printf("+----------+----------------+------------+----------------+----------------+\n");
    }

    return 0;
}