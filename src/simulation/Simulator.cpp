//
// Created by lada on 4/13/24.
//

#include "Simulator.h"
#include "simulation/CellState.h"
#include <omp.h>


Simulator::Simulator(WorldState &worldState) : worldState{worldState} {
    auto &map = worldState.getMap();
    auto &cells = map.getCells();

    std::vector<std::pair<int, int>> coords{
            {20, 20},
    };
    for (auto coord: coords) {
        int storageCoord = map.getStorageCoord(coord.second, coord.first);
        cells[storageCoord] = CellState(CellState::Type::Cell, 200000);
    }
}


void Simulator::step(const SimulatorOptions &options) {
    transferResources();
    replicateCells();
}

void Simulator::transferResources() {
    auto &map = worldState.getMap();

    auto neighborOffsets = map.getNeighborOffsets();

    // Contains padding [(1, 1), (1, 1)] around the borders for vectorization purposes
    std::vector<uint8_t> &validityMask = map.getValidityMask();
    // Array-of-structs: each element contains the full cell state
    std::vector<CellState> &cells = map.getCells();

    std::pair<int, int> storageDims = map.getStorageDims();

    for (auto offset: neighborOffsets) {
#pragma omp simd collapse(2)
        for (int r = 0; r < storageDims.second; r++) {
            for (int q = 0; q < storageDims.first; q++) {
                int neighborCoord = map.getStorageCoord(r + offset.second, q + offset.first);
                int storageCoord = map.getStorageCoord(r, q);
                
                CellState &pointCell = cells[storageCoord];
                CellState &neighborCell = cells[neighborCoord];

                int moveResource = pointCell.resources > 0 &&
                                   pointCell.type == CellState::Type::Cell && 
                                   neighborCell.type == CellState::Type::Cell &&
                                   validityMask[map.getValidityMaskCoord(r, q)] &&
                                   validityMask[map.getValidityMaskCoord(r + offset.second, q + offset.first)];

                pointCell.resources -= moveResource;
                neighborCell.resources += moveResource;
            }
        }
    }
}

void Simulator::replicateCells() {
    auto &map = worldState.getMap();

    auto neighborOffsets = map.getNeighborOffsets();

    std::vector<uint8_t> &validityMask = map.getValidityMask();
    std::vector<CellState> &cells = map.getCells();

    std::pair<int, int> storageDims = map.getStorageDims();

    for (auto offset: neighborOffsets) {
#pragma omp simd collapse(2)
        for (int r = 0; r < storageDims.second; r++) {
            for (int q = 0; q < storageDims.first; q++) {
                int neighborCoord = map.getStorageCoord(r + offset.second, q + offset.first);
                int storageCoord = map.getStorageCoord(r, q);

                CellState &pointCell = cells[storageCoord];
                CellState &neighborCell = cells[neighborCoord];

                int canReplicate = pointCell.resources > 0 &&
                                   pointCell.type == CellState::Type::Cell && 
                                   neighborCell.type == CellState::Type::Air &&
                                   validityMask[map.getValidityMaskCoord(r, q)] &&
                                   validityMask[map.getValidityMaskCoord(r + offset.second, q + offset.first)];

                pointCell.resources -= canReplicate;
                neighborCell.type = canReplicate ? CellState::Type::Cell : neighborCell.type;
            }
        }
    }

}