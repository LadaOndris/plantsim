#pragma once

#include "simulation/GridTopology.h"
#include "simulation/State.h"
#include "simulation/Options.h"
#include "simulation/ISimulator.h"
#include "simulation/CellState.h"
#include <iostream>

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
    GridTopology topology{state.width, state.height};

    void transferResources() {
        const int width = state.width;
        const int height = state.height;
        const auto& neighborOffsets = topology.neighborOffsets;
        const StorageCoord &storageDims = topology.getStorageDimension();

        backBuffer.resources = state.resources;

        for (int y = 0; y < storageDims.y; y++) {
            for (int x = 0; x < storageDims.x; x++) {
                const StorageCoord storage{.x=x, .y=y};
                const int pointIdx = storage.asFlat(storageDims);

                for (const auto& offset : neighborOffsets) {
                    // Here, we can treat the AxialCoord offset as StorageCoord.
                    const StorageCoord neighbor = storage + StorageCoord{.x=offset.q, .y=offset.r};

                    if (!topology.isValid(neighbor)) {
                        continue;
                    }

                    const int neighborIdx = neighbor.asFlat(storageDims);

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
