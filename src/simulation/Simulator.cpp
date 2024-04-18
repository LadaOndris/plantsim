//
// Created by lada on 4/13/24.
//

#include "Simulator.h"


Simulator::Simulator(WorldState &worldState) : worldState{worldState} {
    auto entity = worldState.getEntity();
    auto cells = entity->getCells();

    cells[0]->resources = 10;
}


void Simulator::step(const SimulatorOptions &options) {
    replicateCells();
}

void Simulator::replicateCells() {
    auto entity = worldState.getEntity();
    auto cells = entity->getCells();

    for (auto cell: cells) {
        if (cell->resources <= 0) {
            continue;
        }
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
    }
}