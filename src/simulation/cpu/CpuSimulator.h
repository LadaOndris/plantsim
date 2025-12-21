#pragma once

#include "simulation/cpu/GridShiftHelper.h"
#include "simulation/cpu/ResourceTransfer.h"
#include "simulation/cpu/RandomNeighborReproduction.h"
#include "simulation/cpu/SoilDiffusion.h"
#include "simulation/GridTopology.h"
#include "simulation/State.h"
#include "simulation/Options.h"
#include "simulation/ISimulator.h"

/**
 * @brief CPU-based simulator implementation using Eigen matrix operations.
 */
class CpuSimulator : public ISimulator {
public:
    explicit CpuSimulator(State initialState, const Options& options) 
        : state(std::move(initialState))
        , backBuffer(state)
        , topology(state.width, state.height)
        , grid(topology)
        , resourceTransfer(grid)
        , reproduction(grid)
        , soilDiffusion(grid, options)
    {
        // Resize back buffer for all fields that get double-buffered
        backBuffer.soilWater.resize(state.soilWater.size());
        backBuffer.soilMineral.resize(state.soilMineral.size());
        backBuffer.plantSugar.resize(state.plantSugar.size());
        backBuffer.plantWater.resize(state.plantWater.size());
        backBuffer.plantMineral.resize(state.plantMineral.size());
        backBuffer.cellTypes.resize(state.cellTypes.size());
    }
    
    const State &getState() const override {
        return state;
    }

    void step(const Options &options) override {
        soilDiffusion.step(state, backBuffer, options);
        resourceTransfer.step(state, backBuffer, options);
        reproduction.step(state, backBuffer, options);
    }

private:
    using MatrixXf = GridShiftHelper::MatrixXf;
    using MatrixXi = GridShiftHelper::MatrixXi;

    State state;
    State backBuffer;
    GridTopology topology;
    GridShiftHelper grid;
    
    // Simulation subsystems
    ResourceTransfer resourceTransfer;
    RandomNeighborReproduction reproduction;
    SoilDiffusion soilDiffusion;
};
