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

    for (auto cell : cellsWithResources) {
        assert(cell->resources > 0);

        auto neighbors = worldState.getMap().getNeighbors(*cell);
        for (auto neighbor: neighbors) {
            if (neighbor->type == Point::Type::Cell &&
                neighbor->resources < cell->resources) {
                neighbor->resources += cell->resources;
                cell->resources = 0;
                entity->updateCellsWithResources(neighbor);
                break;
//                neighbor->resources++;
//                cell->resources--;

            }
        }
        entity->updateCellsWithResources(cell);
    }

}

void Simulator::replicateCells() {
    auto entity = worldState.getEntity();
    auto cellsWithResources = entity->getCellsWithResources();

    for (auto cell : cellsWithResources) {
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