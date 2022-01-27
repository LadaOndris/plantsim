//
// Created by lada on 8/26/21.
//

#ifndef PLANTSIM_WORLDSTATE_H
#define PLANTSIM_WORLDSTATE_H

#include <memory>
#include <array>
#include <boost/scoped_array.hpp>
#include "Point.h"
#include "Entity.h"
#include "Process.h"

class Process;

class WorldState {
public:
    WorldState(int width, int height, std::vector<std::shared_ptr<Process>> processes);
    std::shared_ptr<Point> getPoint(std::size_t x, std::size_t y) const;
    std::shared_ptr<Point> operator[](std::size_t index) const;
    /**
     * Calls each process and gives the process corresponding genes, and
     * the process itself determines what to do.
     */
    void invokeProcesses();

    std::shared_ptr<Entity> getEntity();

private:
    std::vector<std::shared_ptr<Point>> points;
    /**
     * An entity contains the processes that each cell
     * can do and a chromosome all cells share.
     */
    std::shared_ptr<Entity> entity;
    /**
     * A cell's behaviour is determined by a vector of processes.
     * Each process is performed only if its corresponding gene dictates it.
     */
    std::vector<std::shared_ptr<Process>> processes;
    int width;
    int height;

    int getTotalGenesCount() const;
};


#endif //PLANTSIM_WORLDSTATE_H
