//
// Created by lada on 4/13/24.
//

#ifndef PLANTSIM_SIMULATOR_H
#define PLANTSIM_SIMULATOR_H


#include "SimulatorOptions.h"
#include "plants/WorldState.h"
#include "simulation/GpuContext.h"
#include <memory>

class Simulator {
public:
    explicit Simulator(WorldState &worldState);

    void step(const SimulatorOptions &options);

    void updateCurrentState();

private:
    void transferResources();

    void replicateCells();

    WorldState &worldState;
    std::unique_ptr<ResourcesSimulator> resourcesSimulator{nullptr};
};





#endif //PLANTSIM_SIMULATOR_H
