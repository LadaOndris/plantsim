#pragma once

#include "simulation/cpu/GridShiftHelper.h"
#include "simulation/State.h"
#include "simulation/Options.h"
#include "simulation/CellState.h"

/**
 * @brief Water and mineral absorption from soil by plant cells.
 * 
 * Implements the uptake mechanism where plant cells absorb water and minerals
 * from overlapping soil at the same grid position:
 * - Plant cells extract resources from the soil layer at their location
 * - Soil loses resources, plant gains them
 * 
 * The uptake is controlled by rates from Options:
 * - waterUptakeRate: Maximum water absorbed per tick
 * - mineralUptakeRate: Maximum minerals absorbed per tick
 * 
 * Uptake is limited by the actual resource available in the soil.
 */
class SoilAbsorption {
public:
    explicit SoilAbsorption(const GridShiftHelper& grid);

    /**
     * @brief Perform one step of soil absorption.
     * 
     * For each plant cell:
     * 1. Calculate the amount of water/minerals to absorb based on uptake rates
     * 2. Limit absorption by what's actually available in the soil at that location
     * 3. Subtract from soil, add to plant
     * 
     * @param state Current simulation state (modified in place)
     * @param backBuffer Buffer for updated state (used for intermediate calculations)
     * @param options Simulation parameters including uptake rates
     */
    void step(State& state, State& backBuffer, const Options& options);

private:
    using MatrixXf = GridShiftHelper::MatrixXf;
    using MatrixXi = GridShiftHelper::MatrixXi;

    const GridShiftHelper& grid;

    MatrixXf plantMask;
    MatrixXf uptakeAmount;

    void applyAbsorption(
        const MatrixXf& soilResource,
        const MatrixXf& plantResource,
        Eigen::Ref<MatrixXf> nextSoilResource,
        Eigen::Ref<MatrixXf> nextPlantResource,
        float uptakeRate,
        float dt
    );
};
