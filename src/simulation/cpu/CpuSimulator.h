#pragma once

#include "simulation/State.h"
#include "simulation/Options.h"
#include "simulation/ISimulator.h"

/**
 * @brief CPU-based simulator implementation.
 */
class CpuSimulator : public ISimulator {
public:
    explicit CpuSimulator(State initialState) : state(std::move(initialState)) {}
    
    const State &getState() const override {
        return state;
    }

    void step(const Options &options) override {
        // TODO: implement.
    }

private:
    State state;
};
