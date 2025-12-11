
# How to run bench apps

The benchmark supports three backends: **CPU**, **SYCL**, and **CUDA**.

## Quick Start

```bash
# Configure with your desired backend
cmake -DBUILD_MAIN=OFF -DBUILD_BENCH=ON -DCMAKE_BUILD_TYPE=Release -DTARGET_BACKEND=<BACKEND> ..

# Build
cmake --build . --target bench_simulation

# Run
./bench/simulation/bench_simulation [steps] [gridSize]
```

## Backend Configuration

### CPU Backend (default)

Uses Eigen with OpenMP for CPU-based simulation.

```bash
cmake -S ~/projects/plantsim/ -B ~/projects/plantsim/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_BENCH=ON \
  -DBUILD_MAIN=OFF \
  -DTARGET_BACKEND=CPU
```

### CUDA Backend

Uses CUDA for NVIDIA GPU acceleration.

```bash
cmake -S ~/projects/plantsim/ -B ~/projects/plantsim/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_CUDA_COMPILER=nvcc \
  -DBUILD_BENCH=ON \
  -DBUILD_MAIN=OFF \
  -DBUILD_TEST=ON \
  -DTARGET_BACKEND=CUDA
```

### SYCL Backend

Uses Intel SYCL (DPC++) for heterogeneous computing.

```bash
cmake -S ~/projects/plantsim/ -B ~/projects/plantsim/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=dpcpp \
  -DBUILD_BENCH=ON \
  -DBUILD_MAIN=OFF \
  -DBUILD_TEST=ON \
  -DTARGET_BACKEND=SYCL
```

## Build and Run

```bash
cmake --build . --target bench_simulation
./bench/simulation/bench_simulation
```

### Example Output

```txt
$ ./bench/sycltest/sycl_test 
Found platform: Intel(R) OpenCL
 Device Intel(R) Corporation 11th Gen Intel(R) Core(TM) i5-11320H @ 3.20GHz
   Global memory size 3.69748 GiB
   Local memory type global
   Local memory size 256 KiB
   Maximum work items dimesions 3
   Maximum work group size 8192
```


## Setup

### Configuring WSL2 for GPU workflows

https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2023-1/configure-wsl-2-for-gpu-workflows.html

Check gpu visibility with: `sycl-ls`


## Testes

```bash
cmake -S ~/projects/plantsim/ -B ~/projects/plantsim/build   -DCMAKE_BUILD_TYPE=Release   -DCMAKE_CXX_COMPILER=dpcpp   -DBUILD_BENCH=OFF   -DBUILD_MAIN=OFF -DBUILD_TEST=ON
```


## Notes on vectorization

https://indico.cern.ch/event/771113/contributions/3203712/attachments/1746730/3022094/PracticalVectorization.pres.pdf

