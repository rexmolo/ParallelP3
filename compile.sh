#!/bin/bash

# Colors for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default flags for CUDA
STD="-std=c++17"
OPT="-O3"
CUDA_FLAGS="-Wno-deprecated-gpu-targets"

# Create directories
mkdir -p bin lib

# Function to compile a specific kernel with main
compile_version() {
    local version=$1
    local kernel_file=$2
    local extra_flags=$3
    
    echo -e "${BLUE}Compiling ${version} version...${NC}"
    
    # Check if kernel file exists
    if [ ! -f "$kernel_file" ]; then
        echo -e "${RED}Kernel file ${kernel_file} not found!${NC}"
        return 1
    fi
    
    # Compile kernel to object file
    echo -e "${BLUE}Compiling kernel ${kernel_file}...${NC}"
    nvcc $STD $OPT $CUDA_FLAGS -c $kernel_file -o lib/${version}.o $extra_flags
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Kernel compilation failed${NC}"
        return 1
    fi
    
    # Compile utils if it exists
    if [ -f "src/utils.cu" ]; then
        nvcc $STD $OPT $CUDA_FLAGS -c src/utils.cu -o lib/utils.o
    fi
    
    # Compile main with conditional compilation and link
    echo -e "${BLUE}Linking with main using conditional compilation...${NC}"
    local kernel_define="-DKERNEL_${version}"
    
    if [ -f "lib/utils.o" ]; then
        nvcc $STD $OPT $CUDA_FLAGS $kernel_define src/main_generic.cu lib/${version}.o lib/utils.o -o bin/${version} -lcublas $extra_flags
    else
        nvcc $STD $OPT $CUDA_FLAGS $kernel_define src/main_generic.cu lib/${version}.o -o bin/${version} -lcublas $extra_flags
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Successfully compiled ${version} to bin/${version}${NC}"
        echo -e "${YELLOW}Run with: ./bin/${version}${NC}"
        return 0
    else
        echo -e "${RED}Linking failed${NC}"
        return 1
    fi
}

# Function to compile all kernels to object files
compile_all_kernels() {
    echo -e "${BLUE}Compiling all kernel files...${NC}"
    
    # Compile each kernel separately
    nvcc $STD $OPT $CUDA_FLAGS -c src/V1_baseline.cu -o lib/V1.o
    nvcc $STD $OPT $CUDA_FLAGS -c src/V2_loopUnroll.cu -o lib/V2.o  
    nvcc $STD $OPT $CUDA_FLAGS -c src/V3_sharedMemory.cu -o lib/V3.o
    nvcc $STD $OPT $CUDA_FLAGS -c src/V4_threadCoarsening.cu -o lib/V4.o
    nvcc $STD $OPT $CUDA_FLAGS -c src/V5_privatization.cu -o lib/V5.o
    nvcc $STD $OPT $CUDA_FLAGS -c src/V6_cublas.cu -o lib/V6.o -lcublas
    if [ -f "src/utils.cu" ]; then
        nvcc $STD $OPT $CUDA_FLAGS -c src/utils.cu -o lib/utils.o
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}All kernels compiled successfully${NC}"
        return 0
    else
        echo -e "${RED}Kernel compilation failed${NC}"
        return 1
    fi
}

# Function to compile main with all kernels
compile_main_all() {
    local output=$1
    local extra_flags=$2
    
    echo -e "${BLUE}Compiling main with all kernels...${NC}"
    
    # Link all object files
    local obj_files="lib/V1.o lib/V2.o lib/V3.o lib/V4.o lib/V5.o lib/V6.o"
    if [ -f "lib/utils.o" ]; then
        obj_files="$obj_files lib/utils.o"
    fi
    
    nvcc $STD $OPT $CUDA_FLAGS src/main.cu $obj_files -o $output -lcublas $extra_flags
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Successfully compiled to ${output}${NC}"
        echo -e "${YELLOW}Run with: ${output}${NC}"
        return 0
    else
        echo -e "${RED}Main compilation failed${NC}"
        return 1
    fi
}

# Process command line arguments
if [ $# -eq 0 ]; then
    echo "Usage: ./compile.sh [option]"
    echo "Options:"
    echo "  V1          - Compile V1 baseline version"
    echo "  V2          - Compile V2 loop unroll version"
    echo "  V3          - Compile V3 shared memory version"
    echo "  V4          - Compile V4 thread coarsening version"
    echo "  V5          - Compile V5 privatization version"
    echo "  V6          - Compile V6 cuBLAS version"
    echo "  all_kernels - Compile all kernels to object files"
    echo "  main_all    - Compile main with all kernels (requires all_kernels first)"
    echo "  all         - Compile all individual versions"
    echo "  clean       - Clean compiled files"
    exit 0
fi

case "$1" in
    V1|v1)
        compile_version "V1" "src/V1_baseline.cu"
        ;;
    V2|v2)
        compile_version "V2" "src/V2_loopUnroll.cu"
        ;;
    V3|v3)
        compile_version "V3" "src/V3_sharedMemory.cu"
        ;;
    V4|v4)
        compile_version "V4" "src/V4_threadCoarsening.cu"
        ;;
    V5|v5)
        compile_version "V5" "src/V5_privatization.cu"
        ;;
    V6|v6)
        compile_version "V6" "src/V6_cublas.cu" "-lcublas"
        ;;
    all_kernels)
        compile_all_kernels
        ;;
    main_all)
        if [ ! -f "lib/V1.o" ] || [ ! -f "lib/V2.o" ]; then
            echo -e "${RED}Kernels not compiled yet. Run './compile.sh all_kernels' first${NC}"
            exit 1
        fi
        compile_main_all bin/main_all
        ;;
    all)
        echo -e "${BLUE}Compiling all individual versions...${NC}"
        compile_version "V1" "src/V1_baseline.cu" && \
        compile_version "V2" "src/V2_loopUnroll.cu" && \
        compile_version "V3" "src/V3_sharedMemory.cu" && \
        compile_version "V4" "src/V4_threadCoarsening.cu" && \
        compile_version "V5" "src/V5_privatization.cu" && \
        compile_version "V6" "src/V6_cublas.cu" "-lcublas"
        echo -e "${GREEN}All versions compiled!${NC}"
        ;;
    clean)
        echo -e "${YELLOW}Cleaning compiled files...${NC}"
        rm -rf bin/* lib/*
        echo -e "${GREEN}Clean completed${NC}"
        ;;
    *)
        echo "Unknown option: $1"
        echo "Use one of: V1, V2, V3, V4, V5, V6, all_kernels, main_all, all, clean"
        exit 1
        ;;
esac

exit 0