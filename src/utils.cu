#include <cstdio>
#include "kernels.cuh"

// Utility functions shared across all kernels

void printVersionTitle(const char* version) {
    printf("\nRunning %s\n", version);
}

void printPerformanceHeader() {
    printf("+----------------+------------+----------------+----------------+\n");
    printf("| Matrix Size    | Block Size | Time (ms)      | Performance    |\n");
    printf("|                |            |                | (GFLOPS)       |\n");
    printf("+----------------+------------+----------------+----------------+\n");
}

void printPerformanceRow(int N, int blockSize, double time_ms, double gflops) {
    printf("| %4d x %4d     | %2d x %2d    | %10.3f     | %10.2f     |\n", 
           N, N, blockSize, blockSize, time_ms, gflops);
}

void printTableFooter() {
    printf("+----------------+------------+----------------+----------------+\n");
}