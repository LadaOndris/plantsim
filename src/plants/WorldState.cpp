//
// Created by lada on 8/26/21.
//

#include "WorldState.h"

#include <utility>
#include "AxialRectangularMap.h"

WorldState::WorldState(std::shared_ptr<Map> map, std::vector<std::shared_ptr<Process>> processes)
        : map(std::move(map)), processes(std::move(processes)) {

    int genesCount = getTotalGenesCount();
    throw std::logic_error("Not implemented");
//    entity = std::make_unique<Entity>(genesCount);
}

void WorldState::invokeProcesses() {
    for (auto &process: processes) {
        for (auto &cell: entity->getCells()) {
            process->invoke(*entity, cell);
        }
    }
}

/*
 * A process requires certain signals to be activated in order
 * for the process to perform.
 * The certain signals are encoded as genes.
 * Thus, each process has to define how much deciding signals it needs.
 */
int WorldState::getTotalGenesCount() const {
    int totalGenesCount = 0;
    throw std::logic_error("Not Implemented");
    //    for (const auto& process : processes) {
//        int genesCount = process->getGenesCount();
//        totalGenesCount += genesCount;
//    }
    return totalGenesCount;
}

std::shared_ptr<Entity> WorldState::getEntity() {
    return entity;
}


