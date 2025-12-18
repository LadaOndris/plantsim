
#pragma once

#include "simulation/State.h"
#include "simulation/Options.h"
#include "simulation/ISimulator.h"
#include "simulation/cuda/CudaState.h"

/**
 * @brief CUDA-based GPU simulator implementation.
 */
class ResourceTransfer  {
public:
    explicit ResourceTransfer(CudaStatePtr initialState);
    ~ResourceTransfer();

    ResourceTransfer(const ResourceTransfer&) = delete;
    ResourceTransfer& operator=(const ResourceTransfer&) = delete;
    
    void step(const Options& options);

private:
    CudaStatePtr ptrCudaState;
    
    // Device memory pointers
    float* d_nextResources = nullptr;
    
    // Pre-allocated device buffers for computation
    float* d_totalOutgoing = nullptr;
    float* d_flowPerNeighbor = nullptr;
    
    void allocateDeviceMemory();
    void freeDeviceMemory();
    void transferResources();
};
