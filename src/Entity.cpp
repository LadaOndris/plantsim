//
// Created by lada on 8/27/21.
//

#include "Entity.h"

#include <utility>


Entity::Entity(std::vector<std::unique_ptr<Process>> processes)
: processes(std::move(processes)), cells() {
    initChromosome();
}

void Entity::invokeProcesses(WorldState& worldState) {
    for (std::unique_ptr<Process> &process : processes) {
        for (std::shared_ptr<Cell> &cell : cells) {
            process->invoke(worldState, cell);
        }
    }
}


int Entity::getTotalGenesCount() const {
    int totalGenesCount = 0;
    for (const std::unique_ptr<Process> &process : processes) {
        int genesCount = process->getGenesCount();
        totalGenesCount += genesCount;
    }
    return totalGenesCount;
}

void Entity::initChromosome() {
    int totalGenesCount = getTotalGenesCount();
    chromosome = std::make_shared<Chromosome<int>>(totalGenesCount);
}

void Entity::addCell(const std::shared_ptr<Cell>& cell) {
    cells.push_back(cell);
}

bool Entity::removeCell(const std::shared_ptr<Cell> &cellToRemove) {
    for(auto it = cells.begin(); it != cells.end(); ++it) {
        std::shared_ptr<Cell> cell = *it;
        if (cell == cellToRemove) {
            cells.erase(it);
            return true;
        }
    }
    return false;
}



