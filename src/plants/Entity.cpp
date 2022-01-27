//
// Created by lada on 8/27/21.
//

#include <utility>
#include "Entity.h"
#include "Process.h"


Entity::Entity(int genesCount) :  cells() {
    initChromosome(genesCount);
}

void Entity::initChromosome(int genesCount) {
    chromosome = std::make_shared<Chromosome<int>>(genesCount);
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

std::vector<std::shared_ptr<Cell>> Entity::getCells() const {
    return cells;
}



