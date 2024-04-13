//
// Created by lada on 8/27/21.
//

#include <utility>
#include "Entity.h"
#include "Process.h"


Entity::Entity(unsigned int NOptions, unsigned int NHormones, unsigned int NResources) :  cells() {
    initChromosome(NOptions, NHormones, NResources);
}

void Entity::initChromosome(unsigned int NOptions, unsigned int NHormones, unsigned int NResources) {
    chromosome = std::make_shared<EntityChromosome>(NOptions, NHormones, NResources);
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

std::shared_ptr<EntityChromosome> Entity::getChromosome() const {
    return chromosome;
}



