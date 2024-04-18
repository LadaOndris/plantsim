//
// Created by lada on 8/27/21.
//

#include <utility>
#include "Entity.h"
#include "Process.h"


Entity::Entity(unsigned int NOptions, unsigned int NHormones, unsigned int NResources) : cells{} {
    //initChromosome(NOptions, NHormones, NResources);
}

void Entity::initChromosome(unsigned int NOptions, unsigned int NHormones, unsigned int NResources) {
    chromosome = std::make_shared<EntityChromosome>(NOptions, NHormones, NResources);
}


void Entity::addCell(Point *cell) {
    cells.push_back(cell);
}

bool Entity::removeCell(const Point *cellToRemove) {
    for (auto it = cells.begin(); it != cells.end(); ++it) {
        auto &cell = *it;
        if (cell == cellToRemove) {
            cells.erase(it);
            return true;
        }
    }
    return false;
}

std::vector<Point *> &Entity::getCells() {
    return cells;
}

std::shared_ptr<EntityChromosome> Entity::getChromosome() const {
    return chromosome;
}



