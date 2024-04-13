//
// Created by lada on 4/13/24.
//

#ifndef PLANTSIM_SIMULATOR_H
#define PLANTSIM_SIMULATOR_H


#include "SimulatorOptions.h"
#include "plants/WorldState.h"

class Simulator {
public:
    explicit Simulator(WorldState &worldState);

    void step(const SimulatorOptions &options);

private:
    WorldState &worldState;
};


#endif //PLANTSIM_SIMULATOR_H
