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

#include "simulation/Options.h"
#include "simulation/CellState.h"
#include "plants/WorldState.h"
#include "sycl/sycl.hpp"
#include <vector>
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


Simulator::Simulator(WorldState &worldState) : worldState{worldState} {
    auto &map = worldState.getMap();
    auto [width, height] = map.getStorageDims();

    // Add a cell with resources for testing
    auto &cells = map.getCells();
    const size_t totalCells = width * height;

    std::vector<std::pair<int, int>> coords{
            {20, 20},
    };
    for (auto coord: coords) {
        int storageCoord = map.getStorageCoord(coord.second, coord.first);
        cells[storageCoord] = CellState(CellState::Type::Cell, 200000);
    }

    // Initialize simulator
    std::vector<int> resourcesHost(totalCells, 0);
    std::vector<int> cellTypesHost(totalCells, 0);
    auto neighborOffsetsHost = std::vector<std::pair<int, int>>(
            map.getNeighborOffsets().begin(),
            map.getNeighborOffsets().end()
    );
    
    // Initialize host data
    for (size_t i = 0; i < totalCells; ++i) {
        resourcesHost[i] = cells[i].resources;
        cellTypesHost[i] = static_cast<int>(cells[i].type);
    }

    resourcesSimulator = std::make_unique<ResourcesSimulator>(
            sycl::queue{sycl::gpu_selector_v},
            width,
            height,
            resourcesHost,
            cellTypesHost,
            neighborOffsetsHost
    );
}


void Simulator::step(const Options &options) {
    resourcesSimulator->step();
}

void Simulator::updateCurrentState() {
    resourcesSimulator->ctx.q.wait();

    auto &map = worldState.getMap();
    auto &cells = map.getCells();
    const size_t totalCells = cells.size();

    // Copy resources back to host
    auto resourcesHost = resourcesSimulator->copyResourcesToHost();

    // Update cell states with new resources
    for (size_t i = 0; i < totalCells; ++i) {
        cells[i].resources = resourcesHost[i];
        cells[i].type = resourcesHost[i] > 0 ? CellState::Type::Cell : CellState::Type::Air;
    }
}


void Simulator::replicateCells() {
    auto &map = worldState.getMap();

    auto neighborOffsets = map.getNeighborOffsets();

    std::pair<int, int> storageDims = map.getStorageDims();

    for (int r = 0; r < storageDims.second; r++) {
        for (int q = 0; q < storageDims.first; q++) {
            for (auto offset: neighborOffsets) {
                CellState &pointCell = map.getCellAt(r, q);
                CellState &neighborCell = map.getCellAt(r + offset.second, q + offset.first);

                int canReplicate = pointCell.resources > 0 &&
                                   pointCell.type == CellState::Type::Cell && 
                                   neighborCell.type == CellState::Type::Air &&
                                   map.isValid(r, q) &&
                                   map.isValid(r + offset.second, q + offset.first);

                pointCell.resources -= canReplicate;
                neighborCell.type = canReplicate ? CellState::Type::Cell : neighborCell.type;
            }
        }
    }
}
