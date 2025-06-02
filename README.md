# CUDA Matrix Multiplication Performance Analysis

## Compile

### Make the compile script executable
```bash
chmod +x compile.sh
```

### Compile Individual Versions

Compile a specific kernel version:
```bash
# Compile V0 (cuBLAS reference)
./compile.sh V0

# Compile V1 (baseline implementation)
./compile.sh V1

# Compile V2 (loop unrolling)
./compile.sh V2

# Compile V3 (shared memory tiling)
./compile.sh V3

# Compile V4 (thread coarsening)
./compile.sh V4

# Compile V5 (privatization/register tiling)
./compile.sh V5

# Compile V6 (final optimized version)
./compile.sh V6_final
```

### Clean Build Files
```bash
./compile.sh clean
```

## Run

### Run Individual Versions

After compiling, executables are created in the `bin/` directory:

```bash
# Run specific version
./bin/V1
./bin/V2
./bin/V3
./bin/V4
./bin/V5
./bin/V6_final

# Run cuBLAS reference
./bin/V0
```

## File Structure
```
ParallelP3/
├── compile.sh          # Compilation script
├── README.md          # This file
├── src/               # Source files
│   ├── main.cu
│   ├── main_generic.cu
│   ├── V0_reference.cu
│   ├── V1_baseline.cu
│   ├── V2_loopUnroll.cu
│   ├── V3_sharedMemory.cu
│   ├── V4_threadCoarsening.cu
│   ├── V5_privatization.cu
│   └── V6_final.cu
├── bin/               # Compiled executables
```
