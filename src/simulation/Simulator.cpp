//
// Created by lada on 4/13/24.
//

#include "Simulator.h"
#include <omp.h>


Simulator::Simulator(WorldState &worldState) : worldState{worldState} {
    auto &map = worldState.getMap();
    std::vector<int> &resources = map.getResources();
    std::vector<Point::Type> &types = map.getPointTypes();

    std::vector<std::pair<int, int>> coords{
            {20, 20},
    };
    for (auto coord: coords) {
        int storageCoord = map.getStorageCoord(coord.first, coord.second);
        types[storageCoord] = Point::Type::Cell;
        resources[storageCoord] = 200000;
    }
}


void Simulator::step(const SimulatorOptions &options) {
    transferResources();
    replicateCells();
}

void Simulator::transferResources() {
    auto &map = worldState.getMap();

    auto neighborOffsets = map.getNeighborOffsets();

    // Contains padding [(1, 1), (1, 1)] around the borders vectorization purposes
    std::vector<bool> &validityMask = map.getValidityMask();
    // A matrix of resources per cell. Stored in the same way as storage.
    std::vector<int> &resources = map.getResources();
    std::vector<Point::Type> &pointTypes = map.getPointTypes();

    std::pair<int, int> storageDims = map.getStorageDims();

    for (auto offset: neighborOffsets) {
#pragma omp simd collapse(2)
        for (int r = 0; r < storageDims.second; r++) {
            for (int q = 0; q < storageDims.first; q++) {
                int neighborCoord = map.getStorageCoord(r + offset.second, q + offset.first);
                int storageCoord = map.getStorageCoord(r, q);
                int pointType = pointTypes[storageCoord];
                int neighborType = pointTypes[neighborCoord];

                int *pointResources = &resources[storageCoord];
                int *neighborResources = &resources[neighborCoord];

                int moveResource = *pointResources > 0 &&
                                   pointType == Point::Type::Cell && neighborType == Point::Type::Cell &&
                                   validityMask[map.getValidityMaskCoord(r, q)] &&
                                   validityMask[map.getValidityMaskCoord(r + offset.second, q + offset.first)];

                *pointResources -= moveResource;
                *neighborResources += moveResource;
            }
        }
    }
}

void Simulator::replicateCells() {
    auto &map = worldState.getMap();

    auto neighborOffsets = map.getNeighborOffsets();

    std::vector<bool> &validityMask = map.getValidityMask();
    std::vector<int> &resources = map.getResources();
    std::vector<Point::Type> &pointTypes = map.getPointTypes();

    std::pair<int, int> storageDims = map.getStorageDims();

    for (auto offset: neighborOffsets) {
#pragma omp simd collapse(2)
        for (int r = 0; r < storageDims.second; r++) {
            for (int q = 0; q < storageDims.first; q++) {
                int neighborCoord = map.getStorageCoord(r + offset.second, q + offset.first);
                int storageCoord = map.getStorageCoord(r, q);

                int pointType = pointTypes[storageCoord];
                auto neighborType = &pointTypes[neighborCoord];
                int *pointResources = &resources[storageCoord];

                int canReplicate = *pointResources > 0 &&
                                   pointType == Point::Type::Cell && *neighborType == Point::Type::Air &&
                                   validityMask[map.getValidityMaskCoord(r, q)] &&
                                   validityMask[map.getValidityMaskCoord(r + offset.second, q + offset.first)];

                *pointResources -= canReplicate;
                *neighborType = canReplicate ? static_cast<Point::Type>(Point::Type::Cell) : *neighborType;
            }
        }
    }

}