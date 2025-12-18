#pragma once

#include "simulation/State.h"
#include "simulation/Options.h"
#include "simulation/ISimulator.h"
#include "simulation/cuda/CudaState.h"
#include "simulation/cuda/ResourceTransfer.h"
#include "simulation/cuda/RandomNeighborReproduction.h"

/**
 * @brief CUDA-based GPU simulator implementation.
 */
class CudaSimulator : public ISimulator {
public:
    explicit CudaSimulator(State initialState);

    const State& getState() const override;

    void step(const Options& options) override;

private:
    StatePtr state;
    CudaStatePtr cudaState;

    ResourceTransfer resourceTransfer;
    RandomNeighborReproduction reproduction;
};
