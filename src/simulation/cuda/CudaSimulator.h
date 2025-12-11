#pragma once

#include "simulation/State.h"
#include "simulation/Options.h"
#include "simulation/ISimulator.h"
#include <utility>

/**
 * @brief CUDA-based GPU simulator implementation.
 */
class CudaSimulator : public ISimulator {
public:
    explicit CudaSimulator(State initialState);
    ~CudaSimulator();

    CudaSimulator(const CudaSimulator&) = delete;
    CudaSimulator& operator=(const CudaSimulator&) = delete;
    
    CudaSimulator(CudaSimulator&& other) noexcept;
    CudaSimulator& operator=(CudaSimulator&& other) noexcept;

    const State& getState() const override;
    void step(const Options& options) override;

private:
    State state;
    
    int storageWidth = 0;
    int storageHeight = 0;
    size_t totalStorageCells = 0;
    
    // Device memory pointers
    float* d_resources = nullptr;
    float* d_nextResources = nullptr;
    int* d_cellTypes = nullptr;
    
    // Pre-allocated device buffers for computation
    float* d_receiverMask = nullptr;
    float* d_neighborsCount = nullptr;
    float* d_totalOutgoing = nullptr;
    float* d_flowPerNeighbor = nullptr;
    float* d_totalIncoming = nullptr;
    
    void allocateDeviceMemory();
    void freeDeviceMemory();
    void copyStateToDevice();
    void copyStateFromDevice();
    void transferResources();
};
