#pragma once

#include "simulation/GridTopology.h"
#include "simulation/State.h"
#include "simulation/Options.h"
#include "simulation/ISimulator.h"
#include "simulation/CellState.h"
#include <omp.h>

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
        const StorageCoord storageDims = topology.getStorageDimension();
        const int storageWidth = storageDims.x;
        const int storageHeight = storageDims.y;
        const int totalCells = storageWidth * storageHeight;

        backBuffer.resources = state.resources;

        const int* __restrict srcResources = state.resources.data();
        const int* __restrict srcCellTypes = state.cellTypes.data();
        int* __restrict dstResources = backBuffer.resources.data();

        for (const auto& offset : topology.neighborOffsets) {
            const int dq = offset.q;
            const int dr = offset.r;

            for (int y = 0; y < storageDims.y; y++) {
                #pragma omp simd
                for (int x = 0; x < storageDims.x; x++) {
                    const StorageCoord storage{.x=x, .y=y};
                    const int idx = storage.asFlat(storageDims);

                    const int nx = x + dq;
                    const int ny = y + dr;

                    // Compute validity masks
                    const int valid = topology.isValid(StorageCoord{.x=x, .y=y}) ? 1 : 0;
                    const int neighborValid = topology.isValid(StorageCoord{.x=nx, .y=ny}) ? 1 : 0;

                    // Safe index: use 0 when invalid to avoid out-of-bounds access
                    const int safeNeighborIdx = neighborValid * (ny * storageWidth + nx);

                    // Transfer conditions as mask (0 or 1)
                    const int hasResources = (srcResources[idx] > 0) ? 1 : 0;
                    const int sourceIsCell = (srcCellTypes[idx] == static_cast<int>(CellState::Type::Cell)) ? 1 : 0;
                    const int neighborIsCell = (srcCellTypes[safeNeighborIdx] == static_cast<int>(CellState::Type::Cell)) ? 1 : 0;

                    const int canTransfer = valid * neighborValid * hasResources * sourceIsCell * neighborIsCell;

                    dstResources[idx] -= canTransfer;
                    dstResources[safeNeighborIdx] += canTransfer;
                }
            }
        }

        std::swap(state.resources, backBuffer.resources);
    }
};
