//
// Created by lada on 8/26/21.
//

#ifndef PLANTSIM_WORLDSTATE_H
#define PLANTSIM_WORLDSTATE_H

#include <memory>
#include <array>
#include "Point.h"
#include "Entity.h"
#include "Process.h"
#include "Map.h"
#include "AxialRectangularMap.h"

class Process;

class WorldState {
public:
    WorldState(std::shared_ptr<AxialRectangularMap> map, std::vector<std::shared_ptr<Process>> processes);

    /**
     * Calls each process and gives the process corresponding genes, and
     * the process itself determines what to do.
     */
    void invokeProcesses();

    std::shared_ptr<Entity> getEntity();

    [[nodiscard]] AxialRectangularMap &getMap() const {
        return *map;
    }

    int getTotalGenesCount() const;

private:
    /**
     * Representation of the hexagonal lattice providing
     * an interface to access the points on the lattice.
     */
    std::shared_ptr<AxialRectangularMap> map;

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

};


#endif //PLANTSIM_WORLDSTATE_H
