//
// Created by lada on 8/27/21.
//

#ifndef PLANTSIM_ENTITY_H
#define PLANTSIM_ENTITY_H

#include <memory>
#include "Process.h"
#include "WorldState.h"
#include "Cell.h"

class WorldState;

class Entity {
public:
    /**
     * Initializes a new entity class.
     * Registers given processes that go on inside a cell.
     * Set ups a corresponding chromosome from the defined processes.
     */
    Entity(std::vector<std::unique_ptr<Process>> processes);

    /**
     * Calls each process and gives the process corresponding genes, and
     * the process itself determines what to do.
     */
    void invokeProcesses(WorldState &worldState);

    void addCell(const std::shared_ptr<Cell> &cell);

    /**
     * Removes a cell.
     *
     * @param cell A cell to be removed.
     * @return Returns 'true' if the cell was removed.
     * Returns 'false' if no such cell was found.
     */
    bool removeCell(const std::shared_ptr<Cell> &cell);

private:
    /**
     * A chromosome is a composition of genes, which dictate
     * what processes should be performed.
     */
    std::shared_ptr<Chromosome<int>> chromosome;
    /**
     * A cell's behaviour is determined by a vector of processes.
     * Each process is performed only if its corresponding gene dictates it.
     */
    std::vector<std::unique_ptr<Process>> processes;
    /**
     * Cells of this entity.
     * They interact with each other through processes.
     */
    std::vector<std::shared_ptr<Cell>> cells;

    int getTotalGenesCount() const;

    void initChromosome();
};


#endif //PLANTSIM_ENTITY_H
