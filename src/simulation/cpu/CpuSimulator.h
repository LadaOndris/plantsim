#pragma once

#include "simulation/cpu/utils/GridShiftHelper.h"
#include "simulation/cpu/stages/ResourceTransfer.h"
#include "simulation/cpu/stages/RandomNeighborReproduction.h"
#include "simulation/cpu/stages/SoilDiffusion.h"
#include "simulation/cpu/stages/SoilAbsorption.h"
#include "simulation/cpu/stages/LightComputation.h"
#include "simulation/cpu/stages/Photosynthesis.h"
#include "simulation/cpu/stages/MaintenanceAndDeath.h"
#include "simulation/cpu/stages/DeadDecay.h"
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
        , soilAbsorption(grid)
        , photosynthesis(topology)
        , maintenanceAndDeath(grid)
        , deadDecay(grid)
    {
        // Resize back buffer for all fields that get double-buffered
        backBuffer.soilWater.resize(state.soilWater.size());
        backBuffer.soilMineral.resize(state.soilMineral.size());
        backBuffer.plantSugar.resize(state.plantSugar.size());
        backBuffer.plantWater.resize(state.plantWater.size());
        backBuffer.plantMineral.resize(state.plantMineral.size());
        backBuffer.plantHealth.resize(state.plantHealth.size());
        backBuffer.deadWater.resize(state.deadWater.size());
        backBuffer.deadMineral.resize(state.deadMineral.size());
        backBuffer.cellTypes.resize(state.cellTypes.size());
    }
    
    const State &getState() const override {
        return state;
    }

    void step(const Options &options) override {
        LightComputation::compute(state, options);
        soilDiffusion.step(state, backBuffer, options);
        soilAbsorption.step(state, backBuffer, options);
        
        photosynthesis.apply(state, options);
        resourceTransfer.step(state, backBuffer, options);
        
        maintenanceAndDeath.step(state, backBuffer, options);
        deadDecay.step(state, backBuffer, options);
        
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
    SoilAbsorption soilAbsorption;
    Photosynthesis photosynthesis;
    MaintenanceAndDeath maintenanceAndDeath;
    DeadDecay deadDecay;
};
