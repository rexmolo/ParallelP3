#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include "kernels.cuh"

void runBenchmark(int N, int blockSize) {
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

    // Conditional compilation for different versions
    #ifdef KERNEL_V1
        // Warmup
        runV1Baseline(d_A, d_B, d_C, N, blockSize);
        
        auto start = std::chrono::high_resolution_clock::now();
        runV1Baseline(d_A, d_B, d_C, N, blockSize);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = 2.0 * N * N * N / (time_ms / 1000.0) / 1e9;
        printPerformanceRow(N, blockSize, time_ms, gflops);
        
    #elif defined(KERNEL_V2)
        // Warmup
        runV2LoopUnroll(d_A, d_B, d_C, N, blockSize);
        
        auto start = std::chrono::high_resolution_clock::now();
        runV2LoopUnroll(d_A, d_B, d_C, N, blockSize);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = 2.0 * N * N * N / (time_ms / 1000.0) / 1e9;
        printPerformanceRow(N, blockSize, time_ms, gflops);
        
    #elif defined(KERNEL_V3)
        // Warmup
        runV3SharedMemory(d_A, d_B, d_C, N, blockSize);
        
        auto start = std::chrono::high_resolution_clock::now();
        runV3SharedMemory(d_A, d_B, d_C, N, blockSize);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = 2.0 * N * N * N / (time_ms / 1000.0) / 1e9;
        printPerformanceRow(N, blockSize, time_ms, gflops);
        
    #elif defined(KERNEL_V4)
        // Warmup
        runV4ThreadCoarsening(d_A, d_B, d_C, N, blockSize);
        
        auto start = std::chrono::high_resolution_clock::now();
        runV4ThreadCoarsening(d_A, d_B, d_C, N, blockSize);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = 2.0 * N * N * N / (time_ms / 1000.0) / 1e9;
        printPerformanceRow(N, blockSize, time_ms, gflops);
        
    #elif defined(KERNEL_V5)
        // Warmup
        runV5Privatization(d_A, d_B, d_C, N, blockSize);
        
        auto start = std::chrono::high_resolution_clock::now();
        runV5Privatization(d_A, d_B, d_C, N, blockSize);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double gflops = 2.0 * N * N * N / (time_ms / 1000.0) / 1e9;
        printPerformanceRow(N, blockSize, time_ms, gflops);
        
    #elif defined(KERNEL_V6)
        double time_ms, gflops;
        runV6CuBLAS(d_A, d_B, d_C, N, time_ms, gflops);
        printPerformanceRow(N, blockSize, time_ms, gflops);
        
    #else
        printf("No kernel version specified! Use -DKERNEL_V1, -DKERNEL_V2, etc.\n");
        return;
    #endif

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

    // Print version title once at the beginning
    #ifdef KERNEL_V1
        printVersionTitle("V1 Baseline Kernel");
    #elif defined(KERNEL_V2)
        printVersionTitle("V2 Loop Unroll Kernel");
    #elif defined(KERNEL_V3)
        printVersionTitle("V3 Shared Memory Kernel");
    #elif defined(KERNEL_V4)
        printVersionTitle("V4 Thread Coarsening Kernel");
    #elif defined(KERNEL_V5)
        printVersionTitle("V5 Privatization Kernel");
    #elif defined(KERNEL_V6)
        printVersionTitle("V6 cuBLAS Kernel");
    #endif

    printPerformanceHeader();

    std::vector<int> matrixSizes = {512, 1024, 2048};
    std::vector<int> blockSizes = {16, 32};

    for (int N : matrixSizes) {
        for (int blockSize : blockSizes) {
            if (blockSize * blockSize <= prop.maxThreadsPerBlock) {
                runBenchmark(N, blockSize);
            }
        }
    }

    printTableFooter();

    return 0;
}