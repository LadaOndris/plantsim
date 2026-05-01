#pragma once

#include "ISimulator.h"
#include "Options.h"
#include "State.h"
#include <memory>

class SimulatorFactory {
public:
    static std::unique_ptr<ISimulator> create(State initialState, const Options& options);
    static const char* getBackendName();
};
