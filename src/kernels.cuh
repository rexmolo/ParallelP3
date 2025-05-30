#ifndef KERNELS_CUH
#define KERNELS_CUH

// Kernel function declarations
__global__ void V1_baselineKernel(const float* A, const float* B, float* C, int N);
__global__ void V2_loopUnrollKernel(const float* A, const float* B, float* C, int N);
__global__ void V3_sharedMemoryKernel(const float* A, const float* B, float* C, int N);
__global__ void V4_threadCoarseningKernel(const float* A, const float* B, float* C, int N);
__global__ void V5_privatizationKernel(const float* A, const float* B, float* C, int N);

// Wrapper functions (callable from host)
void runV0Reference(const float* d_A, const float* d_B, float* d_C, int N, double& time_ms, double& gflops);
void runV1Baseline(const float* d_A, const float* d_B, float* d_C, int N, int blockSize);
void runV2LoopUnroll(const float* d_A, const float* d_B, float* d_C, int N, int blockSize);
void runV3SharedMemory(const float* d_A, const float* d_B, float* d_C, int N, int blockSize);
void runV4ThreadCoarsening(const float* d_A, const float* d_B, float* d_C, int N, int blockSize);
void runV5Privatization(const float* d_A, const float* d_B, float* d_C, int N, int blockSize);
void runV6Final(const float* d_A, const float* d_B, float* d_C, int N, int blockSize);

// Utility functions
void printVersionTitle(const char* version);
void printPerformanceHeader();
void printPerformanceRow(int N, int blockSize, double time_ms, double gflops);
void printTableFooter();
#endif // KERNELS_CUH