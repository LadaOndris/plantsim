
#pragma once

#include "simulation/Options.h"
#include "simulation/GpuContext.h"
#include "plants/WorldState.h"
#include <memory>

class Simulator {
public:
    explicit Simulator(WorldState &worldState);

    void step(const Options &options);

    void updateCurrentState();

private:
    void transferResources();

    void replicateCells();

    WorldState &worldState;
    std::unique_ptr<ResourcesSimulator> resourcesSimulator{nullptr};
};
