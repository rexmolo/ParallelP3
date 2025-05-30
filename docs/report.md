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

# V1-- baseline

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

+-----------+------------+----------------+----------------+----------------+
| Matrix    | Block Size | Threads/Block  | Time (ms)      | Performance    |
| Size      | (threads)  | Total          |                | (GFLOPS)       |
+-----------+------------+----------------+----------------+----------------+
|  256 x 256 |  8 x  8    |    65536       |      0.213     |     157.79     |
|  256 x 256 | 16 x 16    |    65536       |      0.159     |     211.51     |
|  256 x 256 | 32 x 32    |    65536       |      0.180     |     185.92     |
|  512 x 512 |  8 x  8    |   262144       |      1.571     |     170.88     |
|  512 x 512 | 16 x 16    |   262144       |      1.061     |     253.11     |
|  512 x 512 | 32 x 32    |   262144       |      1.089     |     246.48     |
| 1024 x1024 |  8 x  8    |  1048576       |     13.684     |     156.94     |
| 1024 x1024 | 16 x 16    |  1048576       |      9.125     |     235.35     |
| 1024 x1024 | 32 x 32    |  1048576       |      8.753     |     245.35     |
| 2048 x2048 |  8 x  8    |  4194304       |    104.365     |     164.61     |
| 2048 x2048 | 16 x 16    |  4194304       |     61.546     |     279.14     |
| 2048 x2048 | 32 x 32    |  4194304       |     54.112     |     317.49     |
+-----------+------------+----------------+----------------+----------------+


V2-- unroll

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

+----------+----------------+------------+----------------+----------------+
| Version  | Matrix Size    | Block Size | Time (ms)      | Performance    |
|          |                |            |                | (GFLOPS)       |
+----------+----------------+------------+----------------+----------------+
Running V2 Loop Unroll Kernel
| V2_Unroll |  512 x  512     | 16 x 16    |      1.061     |     253.02     |
Running V2 Loop Unroll Kernel
| V2_Unroll |  512 x  512     | 32 x 32    |      1.088     |     246.67     |
+----------+----------------+------------+----------------+----------------+
Running V2 Loop Unroll Kernel
| V2_Unroll | 1024 x 1024     | 16 x 16    |      9.088     |     236.30     |
Running V2 Loop Unroll Kernel
| V2_Unroll | 1024 x 1024     | 32 x 32    |     15.887     |     135.17     |
+----------+----------------+------------+----------------+----------------+
Running V2 Loop Unroll Kernel
| V2_Unroll | 2048 x 2048     | 16 x 16    |     72.972     |     235.43     |
Running V2 Loop Unroll Kernel
| V2_Unroll | 2048 x 2048     | 32 x 32    |     72.903     |     235.65     |
+----------+----------------+------------+----------------+----------------+


V3-- shared memory

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


+----------+----------------+------------+----------------+----------------+
| Version  | Matrix Size    | Block Size | Time (ms)      | Performance    |
|          |                |            |                | (GFLOPS)       |
+----------+----------------+------------+----------------+----------------+
Running V3 Shared Memory Kernel
| V3_Shared |  512 x  512     | 16 x 16    |      0.521     |     515.25     |
Running V3 Shared Memory Kernel
| V3_Shared |  512 x  512     | 32 x 32    |      0.000     | 3677198.03     |
+----------+----------------+------------+----------------+----------------+
Running V3 Shared Memory Kernel
| V3_Shared | 1024 x 1024     | 16 x 16    |      4.183     |     513.43     |
Running V3 Shared Memory Kernel
| V3_Shared | 1024 x 1024     | 32 x 32    |      0.000     | 39045157.24     |
+----------+----------------+------------+----------------+----------------+
Running V3 Shared Memory Kernel
| V3_Shared | 2048 x 2048     | 16 x 16    |     33.255     |     516.60     |
Running V3 Shared Memory Kernel
| V3_Shared | 2048 x 2048     | 32 x 32    |      0.000     | 256415957.97     |
+----------+----------------+------------+----------------+----------------+

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

// Wrapper function for V4
void runV4ThreadCoarsening(const float* d_A, const float* d_B, float* d_C, int N, int blockSize) {
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 blocksPerGrid((N + blockSize * COARSE_FACTOR - 1) / (blockSize * COARSE_FACTOR), 
                       (N + blockSize - 1) / blockSize);
    V4_threadCoarseningKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
}
+----------+----------------+------------+----------------+----------------+
| Version  | Matrix Size    | Block Size | Time (ms)      | Performance    |
|          |                |            |                | (GFLOPS)       |
+----------+----------------+------------+----------------+----------------+
Running V4 Thread Coarsening Kernel
| V4_Coarse |  512 x  512     | 16 x 16    |      2.812     |      95.47     |
Running V4 Thread Coarsening Kernel
| V4_Coarse |  512 x  512     | 32 x 32    |      3.463     |      77.51     |
+----------+----------------+------------+----------------+----------------+
Running V4 Thread Coarsening Kernel
| V4_Coarse | 1024 x 1024     | 16 x 16    |     21.072     |     101.91     |
Running V4 Thread Coarsening Kernel
| V4_Coarse | 1024 x 1024     | 32 x 32    |     26.948     |      79.69     |
+----------+----------------+------------+----------------+----------------+
Running V4 Thread Coarsening Kernel
| V4_Coarse | 2048 x 2048     | 16 x 16    |    185.312     |      92.71     |
Running V4 Thread Coarsening Kernel
| V4_Coarse | 2048 x 2048     | 32 x 32    |    411.498     |      41.75     |
+----------+----------------+------------+----------------+----------------+


V5-- privatization

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
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    V5_privatizationKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
}
+----------+----------------+------------+----------------+----------------+
| Version  | Matrix Size    | Block Size | Time (ms)      | Performance    |
|          |                |            |                | (GFLOPS)       |
+----------+----------------+------------+----------------+----------------+
Running V5 Privatization Kernel
| V5_Privat |  512 x  512     | 16 x 16    |      1.953     |     137.47     |
Running V5 Privatization Kernel
| V5_Privat |  512 x  512     | 32 x 32    |      0.000     | 5478274.61     |
+----------+----------------+------------+----------------+----------------+
Running V5 Privatization Kernel
| V5_Privat | 1024 x 1024     | 16 x 16    |     15.137     |     141.87     |
Running V5 Privatization Kernel
| V5_Privat | 1024 x 1024     | 32 x 32    |      0.000     | 52377649.95     |
+----------+----------------+------------+----------------+----------------+
Running V5 Privatization Kernel
| V5_Privat | 2048 x 2048     | 16 x 16    |    109.882     |     156.35     |
Running V5 Privatization Kernel
| V5_Privat | 2048 x 2048     | 32 x 32    |      0.000     | 330382099.69     |
+----------+----------------+------------+----------------+----------------+



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
+----------+----------------+------------+----------------+----------------+
| Version  | Matrix Size    | Block Size | Time (ms)      | Performance    |
|          |                |            |                | (GFLOPS)       |
+----------+----------------+------------+----------------+----------------+
Running V6 cuBLAS Kernel
| V6_cuBLAS |  512 x  512     | 16 x 16    |      2.737     |      98.08     |
Running V6 cuBLAS Kernel
| V6_cuBLAS |  512 x  512     | 32 x 32    |      0.126     |    2137.82     |
+----------+----------------+------------+----------------+----------------+
Running V6 cuBLAS Kernel
| V6_cuBLAS | 1024 x 1024     | 16 x 16    |      1.052     |    2041.81     |
Running V6 cuBLAS Kernel
| V6_cuBLAS | 1024 x 1024     | 32 x 32    |      0.734     |    2925.54     |
+----------+----------------+------------+----------------+----------------+
Running V6 cuBLAS Kernel
| V6_cuBLAS | 2048 x 2048     | 16 x 16    |      5.031     |    3414.59     |
Running V6 cuBLAS Kernel
| V6_cuBLAS | 2048 x 2048     | 32 x 32    |      5.033     |    3413.57     |
+----------+----------------+------------+----------------+----------------+












+----------+----------------+------------+----------------+----------------+
| Version  | Matrix Size    | Block Size | Time (ms)      | Performance    |
|          |                |            |                | (GFLOPS)       |
+----------+----------------+------------+----------------+----------------+
| V1_Base  |  512 x  512     | 16 x 16    |      1.061     |     252.93     |
| V2_Unroll |  512 x  512     | 16 x 16    |      1.058     |     253.67     |
| V4_Coarse |  512 x  512     | 16 x 16    |      2.816     |      95.33     |
| V3_Shared |  512 x  512     | 16 x 16    |      0.524     |     512.22     |
| V6_cuBLAS |  512 x  512     | 16 x 16    |      1.900     |     141.26     |
| V1_Base  |  512 x  512     | 32 x 32    |      1.097     |     244.76     |
| V2_Unroll |  512 x  512     | 32 x 32    |      1.090     |     246.22     |
| V4_Coarse |  512 x  512     | 32 x 32    |      3.469     |      77.39     |
| V6_cuBLAS |  512 x  512     | 32 x 32    |      0.128     |    2100.75     |
+----------+----------------+------------+----------------+----------------+
| V1_Base  | 1024 x 1024     | 16 x 16    |      9.101     |     235.96     |
| V2_Unroll | 1024 x 1024     | 16 x 16    |     11.073     |     193.94     |
| V4_Coarse | 1024 x 1024     | 16 x 16    |     21.069     |     101.92     |
| V3_Shared | 1024 x 1024     | 16 x 16    |      4.179     |     513.90     |
| V6_cuBLAS | 1024 x 1024     | 16 x 16    |      6.211     |     345.74     |
| V1_Base  | 1024 x 1024     | 32 x 32    |      7.954     |     269.99     |
| V2_Unroll | 1024 x 1024     | 32 x 32    |      7.863     |     273.10     |
| V4_Coarse | 1024 x 1024     | 32 x 32    |     25.957     |      82.73     |
| V6_cuBLAS | 1024 x 1024     | 32 x 32    |      0.664     |    3232.40     |
+----------+----------------+------------+----------------+----------------+
| V1_Base  | 2048 x 2048     | 16 x 16    |     63.478     |     270.64     |
| V2_Unroll | 2048 x 2048     | 16 x 16    |     59.720     |     287.67     |
| V4_Coarse | 2048 x 2048     | 16 x 16    |    183.516     |      93.62     |
| V3_Shared | 2048 x 2048     | 16 x 16    |     23.965     |     716.87     |
| V6_cuBLAS | 2048 x 2048     | 16 x 16    |      3.777     |    4549.03     |
| V1_Base  | 2048 x 2048     | 32 x 32    |     54.537     |     315.01     |
| V2_Unroll | 2048 x 2048     | 32 x 32    |     53.736     |     319.71     |
| V4_Coarse | 2048 x 2048     | 32 x 32    |    398.899     |      43.07     |
| V6_cuBLAS | 2048 x 2048     | 32 x 32    |      3.742     |    4591.29     |
+----------+----------------+------------+----------------+----------------+