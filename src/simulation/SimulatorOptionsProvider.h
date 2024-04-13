//
// Created by lada on 4/13/24.
//

#ifndef PLANTSIM_SIMULATOROPTIONSPROVIDER_H
#define PLANTSIM_SIMULATOROPTIONSPROVIDER_H


#include "SimulatorOptions.h"

class SimulatorOptionsProvider {
public:
    virtual SimulatorOptions getSimulatorOptions() const = 0;
};


#endif //PLANTSIM_SIMULATOROPTIONSPROVIDER_H
