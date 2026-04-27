#pragma once

#include "simulation/cpu/utils/GridShiftHelper.h"
#include "simulation/State.h"
#include "simulation/Options.h"
#include "simulation/CellState.h"

/**
 * @brief Simulation stage for plant cell maintenance costs and death.
 * 
 * Each plant cell pays maintenance costs in sugar and water per tick.
 * When resources are insufficient, health is damaged. When health reaches
 * zero, the cell dies and becomes a Dead cell, with resources transferred
 * to dead pools for recycling.
 */
class MaintenanceAndDeath {
public:
    using MatrixXf = GridShiftHelper::MatrixXf;
    using MatrixXi = GridShiftHelper::MatrixXi;

    explicit MaintenanceAndDeath(const GridShiftHelper& grid);

    void step(State& state, State& backBuffer, const Options& options);

private:
    const GridShiftHelper& grid;
    
    MatrixXf plantMask;
    MatrixXf sugarBefore;
    MatrixXf waterBefore;
    MatrixXf sugarDeficit;
    MatrixXf waterDeficit;
    MatrixXf damage;
    MatrixXf lightTerm;
    MatrixXf waterDemand;
};
