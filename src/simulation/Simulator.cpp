//
// Created by lada on 4/13/24.
//

#include "Simulator.h"


Simulator::Simulator(WorldState &worldState) : worldState{worldState} {
    auto entity = worldState.getEntity();
    auto cells = entity->getCells();

    cells[0]->resources = 10000;
    entity->updateCellsWithResources(cells[0]);
}


void Simulator::step(const SimulatorOptions &options) {
    transferResources();
    replicateCells();
}

void Simulator::transferResources() {
    auto entity = worldState.getEntity();
    auto cellsWithResources = entity->getCellsWithResources();

    auto &map = worldState.getMap();
    auto maxCoords = map.getMaxCoords();

    auto neighborOffsets = map.getNeighborOffsets();
    auto offset = neighborOffsets[0];


    // Contains padding around the borders vectorization purposes
    std::vector<bool> &validityMask = map.getValidityMask();
    // A matrix of resources per cell. Stored in the same way as storage.
    std::vector<int> &resources = map.getResources();

    for (int r = 0; r < maxCoords.second; r++) {
        for (int q = 0; q < maxCoords.first; q++) {
            // TODO: can be out of bounds, create a validity mask
            auto cell = map.getPointMaybeInvalid(q, r);
            auto neighbor = map.getPointMaybeInvalid(q + offset.first, r + offset.second);

            int moveResource = cell->resources > 0 && validityMask[q, r] && validityMaskk[q + offset.first, r + offset.second];

            cell->resources -= moveResource;
            neighbor->resources += moveResource;
        }
    }

//    for (auto cell : cellsWithResources) {
//        assert(cell->resources > 0);
//
//        auto neighbors = worldState.getMap().getNeighbors(*cell);
//        for (auto neighbor: neighbors) {
//            if (neighbor->type == Point::Type::Cell &&
//                neighbor->resources < cell->resources) {
//                neighbor->resources += cell->resources;
//                cell->resources = 0;
//                entity->updateCellsWithResources(neighbor);
//                break;
////                neighbor->resources++;
////                cell->resources--;
//
//            }
//        }
//        entity->updateCellsWithResources(cell);
//    }

}

void Simulator::replicateCells() {
    auto entity = worldState.getEntity();
    auto cellsWithResources = entity->getCellsWithResources();

    for (auto cell: cellsWithResources) {
        assert(cell->resources > 0);

        auto neighbors = worldState.getMap().getNeighbors(*cell);
        for (auto neighbor: neighbors) {
            if (cell->resources <= 0) {
                break;
            }
            // Replicate the point to its neighbors
            if (neighbor->type != Point::Type::Cell) {
                neighbor->type = Point::Type::Cell;
                entity->addCell(neighbor);
                cell->resources--;
            }
        }
        entity->updateCellsWithResources(cell);
    }
}