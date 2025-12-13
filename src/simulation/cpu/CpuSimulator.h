#pragma once

#include "simulation/cpu/GridShiftHelper.h"
#include "simulation/cpu/ResourceTransfer.h"
#include "simulation/cpu/RandomNeighborReproduction.h"
#include "simulation/GridTopology.h"
#include "simulation/State.h"
#include "simulation/Options.h"
#include "simulation/ISimulator.h"

/**
 * @brief CPU-based simulator implementation using Eigen matrix operations.
 */
class CpuSimulator : public ISimulator {
public:
    explicit CpuSimulator(State initialState) 
        : state(std::move(initialState))
        , backBuffer(state)
        , topology(state.width, state.height)
        , grid(topology)
        , resourceTransfer(grid)
        , reproduction(grid)
    {}
    
    const State &getState() const override {
        return state;
    }

    void step(const Options &options) override {
        resourceTransfer.step(state, backBuffer, options);
        reproduction.step(state, backBuffer, options);
    }

private:
    State state;
    State backBuffer;
    GridTopology topology;
    GridShiftHelper grid;
    
    // Simulation subsystems
    ResourceTransfer resourceTransfer;
    RandomNeighborReproduction reproduction;
};
