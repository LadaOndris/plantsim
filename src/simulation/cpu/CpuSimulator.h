#pragma once

#include "simulation/State.h"
#include "simulation/Options.h"
#include "simulation/ISimulator.h"
#include "simulation/CellState.h"

/**
 * @brief CPU-based simulator implementation.
 */
class CpuSimulator : public ISimulator {
public:
    explicit CpuSimulator(State initialState) 
        : state(std::move(initialState))
        , backBuffer(state) {}
    
    const State &getState() const override {
        return state;
    }

    void step(const Options &options) override {
        if (options.enableResourceTransfer) {
            transferResources();
        }
    }

private:
    State state;
    State backBuffer;

    void transferResources() {
        const int width = state.width;
        const int height = state.height;
        const auto& neighborOffsets = state.neighborOffsets;

        backBuffer.resources = state.resources;

        for (int r = 0; r < height; r++) {
            for (int q = 0; q < width; q++) {
                const int pointIdx = r * width + q;
                
                for (const auto& offset : neighborOffsets) {
                    const int nq = q + offset.first;
                    const int nr = r + offset.second;
                    
                    if (!state.isValid(nr, nq)) {
                        continue;
                    }
                    
                    const int neighborIdx = nr * width + nq;
                    
                    // Transfer resource if:
                    // - source cell has resources
                    // - source cell is of type Cell
                    // - neighbor cell is of type Cell
                    bool canTransfer = state.resources[pointIdx] > 0 &&
                                       state.cellTypes[pointIdx] == static_cast<int>(CellState::Type::Cell) &&
                                       state.cellTypes[neighborIdx] == static_cast<int>(CellState::Type::Cell);
                    
                    if (canTransfer) {
                        backBuffer.resources[pointIdx] -= 1;
                        backBuffer.resources[neighborIdx] += 1;
                        break;
                    }
                }
            }
        }

        std::swap(state.resources, backBuffer.resources);
    }
};
