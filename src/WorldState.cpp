//
// Created by lada on 8/26/21.
//

#include "WorldState.h"

WorldState::WorldState(int width, int height, std::vector<std::shared_ptr<Process>> processes)
    : processes(std::move(processes)), width(width), height(height) {

    int genesCount = getTotalGenesCount();
    entity = std::make_unique<Entity>(genesCount);


    //scoped_array mapTemp(new Point[width * height]);
    //map.swap(mapTemp);
}

std::shared_ptr<Point> WorldState::getPoint(std::size_t x, std::size_t y) const {
    return (*this)[x * width + y];
}

std::shared_ptr<Point> WorldState::operator[](std::size_t index) const {
    return points[index];
}


void WorldState::invokeProcesses() {
    for (auto& process : processes) {
        for (auto& cell : entity->getCells()) {
            process->invoke(*this, cell);
        }
    }
}

int WorldState::getTotalGenesCount() const {
    int totalGenesCount = 0;
    for (const auto& process : processes) {
        int genesCount = process->getGenesCount();
        totalGenesCount += genesCount;
    }
    return totalGenesCount;
}

std::shared_ptr<Entity> WorldState::getEntity() {
    return entity;
}


