#pragma once

#include "simulation/State.h"
#include "simulation/Options.h"
#include "simulation/ISimulator.h"

/**
 * @brief SYCL-based GPU simulator implementation.
 */
class SyclSimulator : public ISimulator {
public:
    explicit SyclSimulator(State initialState) : state(std::move(initialState)) {}

    const State &getState() const override {
        return state;
    }

    void step(const Options &options) override {
        // TODO: implement
    }

private:
    State state;
};
