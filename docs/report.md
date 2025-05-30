# Device version

CUDA Device: Quadro P4000 Pascal
Compute Capability: 6.1
Max Threads per Block: 1024
| **Parameter**    | **Details**                       |
|------------------|---------------------------------|
| GPU Model        | NVIDIA Quadro P4000 (8 GB VRAM) |
| CUDA Version     | 12.8                            |
| Driver Version   | 550.144.03                      |
| Memory Available | 8 GB total (2 MiB currently used)|

Max threads per block: 1024
Max threads per multiprocessor: 2048
Number of multiprocessors (SMs): 14
Max threads per GPU: 28672

Global memory size: 8106 MB
Shared memory per block: 48 KB
Registers per block: 65536

CUDA Device: Quadro P4000
Compute Capability: 6.1
Max Threads per Block: 1024
Max Block Dimensions: 1024 x 1024 x 64

## Design Decisions Based on Hardware:

| Parameter                | Value                      | Justification                                                 |
| ------------------------ | -------------------------- | ------------------------------------------------------------- |
| Tile size (`TILE_WIDTH`) | 32                         | 32×32×4 bytes = 4 KB per tile per matrix (fits shared memory) |
| Threads per block        | 1024 (32×32)               | Max per block and optimal for warp-based execution            |
| Blocks per grid          | `ceil(N / TILE_WIDTH)`     | Covers entire matrix even for large sizes                     |
| Shared memory usage      | 2×32×32×4 = 8 KB per block | Well within 48 KB limit                                       |
| Registers per thread     | < 64                       | Stay under 65536 registers per block to avoid spilling        |


# V1-- baseline

Running V1 Baseline Kernel
+----------------+------------+----------------+----------------+
| Matrix Size    | Block Size | Time (ms)      | Performance    |
|                |            |                | (GFLOPS)       |
+----------------+------------+----------------+----------------+
|  512 x  512     | 16 x 16    |      1.062     |     252.75     |
|  512 x  512     | 32 x 32    |      1.087     |     246.86     |
| 1024 x 1024     | 16 x 16    |      9.109     |     235.75     |
| 1024 x 1024     | 32 x 32    |      8.779     |     244.63     |
| 2048 x 2048     | 16 x 16    |     72.842     |     235.85     |
| 2048 x 2048     | 32 x 32    |     64.962     |     264.46     |
+----------------+------------+----------------+----------------+


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


V2-- unroll


Running V2 Loop Unroll Kernel
+----------------+------------+----------------+----------------+
| Matrix Size    | Block Size | Time (ms)      | Performance    |
|                |            |                | (GFLOPS)       |
+----------------+------------+----------------+----------------+
|  512 x  512     | 16 x 16    |      1.060     |     253.28     |
|  512 x  512     | 32 x 32    |      1.089     |     246.49     |
| 1024 x 1024     | 16 x 16    |     13.905     |     154.44     |
| 1024 x 1024     | 32 x 32    |      8.738     |     245.78     |
| 2048 x 2048     | 16 x 16    |     69.227     |     248.17     |
| 2048 x 2048     | 32 x 32    |     61.388     |     279.86     |
+----------------+------------+----------------+----------------+

__global__ void V2_loopUnrollKernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        int k = 0;
        
        // Unroll loop by 4 to reduce control divergence
        for (; k <= N - 4; k += 4) {
            sum += A[row * N + k] * B[k * N + col];
            sum += A[row * N + k + 1] * B[(k + 1) * N + col];
            sum += A[row * N + k + 2] * B[(k + 2) * N + col];
            sum += A[row * N + k + 3] * B[(k + 3) * N + col];
        }
        
        // Handle remaining elements
        for (; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        
        C[row * N + col] = sum;
    }
}



V3-- shared memory

Running V3 Shared Memory Kernel
+----------------+------------+----------------+----------------+
| Matrix Size    | Block Size | Time (ms)      | Performance    |
|                |            |                | (GFLOPS)       |
+----------------+------------+----------------+----------------+
|  512 x  512     | 16 x 16    |      0.528     |     507.94     |
|  512 x  512     | 32 x 32    |      0.507     |     529.57     |
| 1024 x 1024     | 16 x 16    |      4.190     |     512.58     |
| 1024 x 1024     | 32 x 32    |      3.912     |     548.89     |
| 2048 x 2048     | 16 x 16    |     33.285     |     516.15     |
| 2048 x 2048     | 32 x 32    |     42.425     |     404.94     |
+----------------+------------+----------------+----------------+
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



V4-- coarse-grained
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


Running V4 Thread Coarsening Kernel
+----------------+------------+----------------+----------------+
| Matrix Size    | Block Size | Time (ms)      | Performance    |
|                |            |                | (GFLOPS)       |
+----------------+------------+----------------+----------------+
|  512 x  512     | 16 x 16    |      2.824     |      95.04     |
|  512 x  512     | 32 x 32    |      3.480     |      77.13     |
| 1024 x 1024     | 16 x 16    |     21.103     |     101.76     |
| 1024 x 1024     | 32 x 32    |     26.715     |      80.38     |
| 2048 x 2048     | 16 x 16    |    187.926     |      91.42     |
| 2048 x 2048     | 32 x 32    |    387.539     |      44.33     |
+----------------+------------+----------------+----------------+

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


V5-- privatization

Running V5 Privatization Kernel
+----------------+------------+----------------+----------------+
| Matrix Size    | Block Size | Time (ms)      | Performance    |
|                |            |                | (GFLOPS)       |
+----------------+------------+----------------+----------------+
|  512 x  512     | 16 x 16    |      2.319     |     115.75     |
|  512 x  512     | 32 x 32    |      1.882     |     142.60     |
| 1024 x 1024     | 16 x 16    |     15.272     |     140.62     |
| 1024 x 1024     | 32 x 32    |     18.299     |     117.35     |
| 2048 x 2048     | 16 x 16    |    107.467     |     159.86     |
| 2048 x 2048     | 32 x 32    |     92.434     |     185.86     |
+----------------+------------+----------------+----------------+


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


# reference point
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
+----------------+------------+----------------+----------------+
| Matrix Size    | Block Size | Time (ms)      | Performance    |
|                |            |                | (GFLOPS)       |
+----------------+------------+----------------+----------------+
|  512 x  512     | 16 x 16    |      2.745     |      97.80     |
|  512 x  512     | 32 x 32    |      0.128     |    2096.45     |
| 1024 x 1024     | 16 x 16    |      1.316     |    1631.29     |
| 1024 x 1024     | 32 x 32    |      0.752     |    2856.12     |
| 2048 x 2048     | 16 x 16    |      5.080     |    3381.93     |
| 2048 x 2048     | 32 x 32    |      5.067     |    3390.78     |
+----------------+------------+----------------+----------------+