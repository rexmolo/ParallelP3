#include <cstdio>
#include "kernels.cuh"

// Utility functions shared across all kernels

void printPerformanceHeader() {
    printf("\n");
    printf("+----------+----------------+------------+----------------+----------------+\n");
    printf("| Version  | Matrix Size    | Block Size | Time (ms)      | Performance    |\n");
    printf("|          |                |            |                | (GFLOPS)       |\n");
    printf("+----------+----------------+------------+----------------+----------------+\n");
}

void printPerformanceRow(const char* version, int N, int blockSize, double time_ms, double gflops) {
    printf("| %-8s | %4d x %4d     | %2d x %2d    | %10.3f     | %10.2f     |\n", 
           version, N, N, blockSize, blockSize, time_ms, gflops);
}
