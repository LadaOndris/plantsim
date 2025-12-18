#pragma once

#include "simulation/Options.h"
#include "simulation/cuda/CudaState.h"
#include <cstdint>

/**
 * @brief CUDA-based implementation of random neighbor reproduction.
 *
 * Implements a three-phase algorithm on the GPU:
 *  1. Intention   – eligible cells choose a random empty neighbor
 *  2. Resolution  – conflicts are resolved so only one parent claims a cell
 *  3. Application – parents pay cost and children are created
 */
class RandomNeighborReproduction {
public:
    struct Config {
        float reproductionThreshold = 1.0f;
        float reproductionCost = 1.0f;
        float childInitialResources = 0.0f;
    };

public:
    explicit RandomNeighborReproduction(CudaStatePtr cudaState);
    ~RandomNeighborReproduction();

    RandomNeighborReproduction(const RandomNeighborReproduction&) = delete;
    RandomNeighborReproduction& operator=(const RandomNeighborReproduction&) = delete;

    /**
     * @brief Execute one reproduction step on the GPU.
     */
    void step(const Options& options);

    void setConfig(const Config& cfg) { config = cfg; }
    [[nodiscard]] const Config& getConfig() const { return config; }

private:
    CudaStatePtr ptrCudaState;

    Config config;

    // Core masks
    float* d_emptyMask = nullptr;
    float* d_emptyNeighborCount = nullptr;
    float* d_eligibleMask = nullptr;

    // Intention phase
    int*   d_chosenDirection = nullptr;   // One direction per cell (-1 if none)

    // Resolution & results
    float* d_childMask  = nullptr;         // Target cells for new children
    float* d_parentCost = nullptr;         // Winning parents (1.0 if paid)

    // Output buffers (ping-pong with CudaState)
    float* d_nextResources = nullptr;
    int*   d_nextCellTypes = nullptr;

    // RNG seed (advanced per step)
    uint32_t rngSeed = 0;

private:
    void allocateDeviceMemory();
    void freeDeviceMemory();

    void runIntentionPhase();
    void runResolutionPhase();
    void runApplicationPhase();
};