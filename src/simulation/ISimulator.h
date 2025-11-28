
#pragma once

#include "simulation/State.h"
#include "simulation/Options.h"

/**
 * @brief Abstract base class for simulators.
 */
class ISimulator {
public:
    virtual ~ISimulator() = default;

    virtual const State &getState() const = 0;

    virtual void step(const Options &options) = 0;
};
