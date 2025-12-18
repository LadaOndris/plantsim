#include "simulation/cuda/RandomNeighborReproduction.h"
#include "simulation/CellState.h"
#include "simulation/cuda/CudaUtils.cuh"

#include <cuda_runtime.h>
#include <stdexcept>

__device__ __forceinline__ uint32_t xorshift(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

__global__ void computeEmptyMaskKernel(
    const int* __restrict__ cellTypes,
    float* __restrict__ emptyMask,
    int width,
    int height
) {
    KERNEL_2D_SETUP(x, y, idx, width, height);
    emptyMask[idx] = (cellTypes[idx] == static_cast<int>(CellState::Type::Air)) ? 1.0f : 0.0f;
}

__global__ void countEmptyNeighborsKernel(
    const float* __restrict__ emptyMask,
    float* __restrict__ emptyNeighborCount,
    int width,
    int height
) {
    KERNEL_2D_SETUP(x, y, idx, width, height);

    float count = 0.0f;
    FOR_EACH_NEIGHBOR(x, y, width, height, nidx, {
        count += emptyMask[nidx];
    });

    emptyNeighborCount[idx] = count;
}

__global__ void computeEligibleMaskKernel(
    const float* __restrict__ resources,
    const int* __restrict__ cellTypes,
    const float* __restrict__ emptyNeighborCount,
    float* __restrict__ eligibleMask,
    RandomNeighborReproduction::Config cfg,
    int width,
    int height
) {
    KERNEL_2D_SETUP(x, y, idx, width, height);

    bool eligible =
        cellTypes[idx] == static_cast<int>(CellState::Type::Cell) &&
        resources[idx] >= cfg.reproductionThreshold &&
        emptyNeighborCount[idx] > 0.0f;

    eligibleMask[idx] = eligible ? 1.0f : 0.0f;
}

__global__ void chooseDirectionKernel(
    const float* __restrict__ eligibleMask,
    const float* __restrict__ emptyMask,
    int* __restrict__ chosenDirection,
    uint32_t seed,
    int width,
    int height
) {
    KERNEL_2D_SETUP(x, y, idx, width, height);

    if (eligibleMask[idx] < 0.5f) {
        chosenDirection[idx] = -1;
        return;
    }

    int availableDirs[HEX_NEIGHBOR_COUNT];
    int count = 0;

    FOR_EACH_NEIGHBOR_WITH_DIR(x, y, width, height, dir, nidx, {
        if (emptyMask[nidx] > 0.5f) {
            availableDirs[count++] = dir;
        }
    });

    if (count == 0) {
        chosenDirection[idx] = -1;
        return;
    }

    uint32_t rng = seed ^ static_cast<uint32_t>(idx);
    int choice = xorshift(rng) % count;
    chosenDirection[idx] = availableDirs[choice];
}

__global__ void resolveConflictsKernel(
    const int* __restrict__ chosenDirection,
    const float* __restrict__ emptyMask,
    float* __restrict__ childMask,
    float* __restrict__ parentCost,
    int width,
    int height
) {
    KERNEL_2D_SETUP(x, y, idx, width, height);

    childMask[idx]  = 0.0f;
    parentCost[idx] = 0.0f;

    if (emptyMask[idx] < 0.5f) {
        return;
    }

    int winningParent = -1;

    FOR_EACH_NEIGHBOR_WITH_DIR(x, y, width, height, dir, nidx, {
        int oppositeDir = (dir + 3) % HEX_NEIGHBOR_COUNT;
        if (chosenDirection[nidx] == oppositeDir) {
            winningParent = nidx;
        }
    });

    if (winningParent >= 0) {
        childMask[idx] = 1.0f;
        parentCost[winningParent] = 1.0f;
    }
}

__global__ void applyReproductionKernel(
    const float* __restrict__ resources,
    const int* __restrict__ cellTypes,
    const float* __restrict__ childMask,
    const float* __restrict__ parentCost,
    float* __restrict__ nextResources,
    int* __restrict__ nextCellTypes,
    RandomNeighborReproduction::Config cfg,
    int width,
    int height
) {
    KERNEL_2D_SETUP(x, y, idx, width, height);

    float res = resources[idx];
    int   type = cellTypes[idx];

    res -= parentCost[idx] * cfg.reproductionCost;

    if (childMask[idx] > 0.5f) {
        type = static_cast<int>(CellState::Type::Cell);
        res  = cfg.childInitialResources;
    }

    nextResources[idx] = res;
    nextCellTypes[idx] = type;
}

RandomNeighborReproduction::RandomNeighborReproduction(CudaStatePtr cudaState)
    : ptrCudaState(cudaState)
{
    rngSeed = 0x12345678u;

    allocateDeviceMemory();
}

RandomNeighborReproduction::~RandomNeighborReproduction() {
    freeDeviceMemory();
}

void RandomNeighborReproduction::allocateDeviceMemory() {
    size_t floatSize = ptrCudaState->totalStorageCells * sizeof(float);
    size_t intSize  = ptrCudaState->totalStorageCells * sizeof(int);

    CUDA_CHECK(cudaMalloc(&d_emptyMask,            floatSize));
    CUDA_CHECK(cudaMalloc(&d_emptyNeighborCount,   floatSize));
    CUDA_CHECK(cudaMalloc(&d_eligibleMask,         floatSize));
    CUDA_CHECK(cudaMalloc(&d_chosenDirection,      intSize));
    CUDA_CHECK(cudaMalloc(&d_childMask,            floatSize));
    CUDA_CHECK(cudaMalloc(&d_parentCost,           floatSize));
    CUDA_CHECK(cudaMalloc(&d_nextResources,        floatSize));
    CUDA_CHECK(cudaMalloc(&d_nextCellTypes,        intSize));
}

void RandomNeighborReproduction::freeDeviceMemory() {
    cudaFree(d_emptyMask);
    cudaFree(d_emptyNeighborCount);
    cudaFree(d_eligibleMask);
    cudaFree(d_chosenDirection);
    cudaFree(d_childMask);
    cudaFree(d_parentCost);
    cudaFree(d_nextResources);
    cudaFree(d_nextCellTypes);
}

void RandomNeighborReproduction::step(const Options& options) {
    if (!options.enableCellMultiplication) {
        return;
    }

    KernelConfig cfg2D(ptrCudaState->storageWidth, ptrCudaState->storageHeight);

    computeEmptyMaskKernel<<<cfg2D.gridSize, cfg2D.blockSize>>>(
        ptrCudaState->d_cellTypes,
        d_emptyMask,
        ptrCudaState->storageWidth,
        ptrCudaState->storageHeight
    );

    countEmptyNeighborsKernel<<<cfg2D.gridSize, cfg2D.blockSize>>>(
        d_emptyMask,
        d_emptyNeighborCount,
        ptrCudaState->storageWidth,
        ptrCudaState->storageHeight
    );

    computeEligibleMaskKernel<<<cfg2D.gridSize, cfg2D.blockSize>>>(
        ptrCudaState->d_resources,
        ptrCudaState->d_cellTypes,
        d_emptyNeighborCount,
        d_eligibleMask,
        config,
        ptrCudaState->storageWidth,
        ptrCudaState->storageHeight
    );

    chooseDirectionKernel<<<cfg2D.gridSize, cfg2D.blockSize>>>(
        d_eligibleMask,
        d_emptyMask,
        d_chosenDirection,
        rngSeed,
        ptrCudaState->storageWidth,
        ptrCudaState->storageHeight
    );

    resolveConflictsKernel<<<cfg2D.gridSize, cfg2D.blockSize>>>(
        d_chosenDirection,
        d_emptyMask,
        d_childMask,
        d_parentCost,
        ptrCudaState->storageWidth,
        ptrCudaState->storageHeight
    );

    applyReproductionKernel<<<cfg2D.gridSize, cfg2D.blockSize>>>(
        ptrCudaState->d_resources,
        ptrCudaState->d_cellTypes,
        d_childMask,
        d_parentCost,
        d_nextResources,
        d_nextCellTypes,
        config,
        ptrCudaState->storageWidth,
        ptrCudaState->storageHeight
    );

    CUDA_CHECK(cudaGetLastError());

    std::swap(ptrCudaState->d_resources, d_nextResources);
    std::swap(ptrCudaState->d_cellTypes, d_nextCellTypes);

    rngSeed += 0x9E3779B9u; // advance seed
}