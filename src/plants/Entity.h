//
// Created by lada on 8/27/21.
//

#ifndef PLANTSIM_ENTITY_H
#define PLANTSIM_ENTITY_H

#include <memory>
#include "Cell.h"
#include "genetics/Chromosome.h"
#include "EntityChromosome.h"

class Entity {
public:
    /**
     * Initializes a new entity class.
     * Registers given processes that go on inside a cell.
     * Set ups a corresponding chromosome from the defined processes.
     */
    explicit Entity(unsigned int NOptions, unsigned int NHormones, unsigned int NResources);

    void addCell(const std::shared_ptr<Cell> &cell);

    /**
     * Removes a cell.
     *
     * @param cell A cell to be removed.
     * @return Returns 'true' if the cell was removed.
     * Returns 'false' if no such cell was found.
     */
    bool removeCell(const std::shared_ptr<Cell> &cell);

    std::vector<std::shared_ptr<Cell>> getCells() const;

    std::shared_ptr<EntityChromosome> getChromosome() const;

private:
    /**
     * A chromosome is a composition of genes, which dictate
     * what processes should be performed.
     */
    std::shared_ptr<EntityChromosome> chromosome;
    /**
     * Cells of this entity.
     * They interact with each other through processes.
     */
    std::vector<std::shared_ptr<Cell>> cells;

    void initChromosome(unsigned NOptions, unsigned NHormones, unsigned NResources);
};


#endif //PLANTSIM_ENTITY_H
