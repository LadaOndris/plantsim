#pragma once

#include "simulation/cpu/GridShiftHelper.h"
#include "simulation/State.h"
#include "simulation/Options.h"
#include "simulation/CellState.h"

/**
 * @brief Nutrient diffusion and absorption system.
 * 
 * Manages a diffusive nutrient field on the hex grid with:
 * - Diffusion: nutrients flow from high to low concentration between neighbors
 * - Absorption: cells deplete local nutrients based on absorption rate
 * - Regeneration: soil layer (bottom rows) regenerates nutrients
 * - Clamping: nutrient values are clamped to [0, maxNutrient]
 */
class NutrientDiffusion {
public:
    explicit NutrientDiffusion(const GridShiftHelper& grid, const Options& options);

    /**
     * @brief Perform one step of nutrient update.
     * 
     * Executes three phases:
     * 1. Diffusion: Laplacian-based diffusion between neighboring cells
     * 2. Absorption: Cells absorb nutrients from their location
     * 3. Regeneration: Soil layer regenerates nutrients toward maxNutrient
     * 
     * @param state Current simulation state (read from)
     * @param backBuffer Buffer for updated state (write to)
     * @param options Simulation parameters
     */
    void step(State& state, State& backBuffer, const Options& options);

private:
    using MatrixXf = GridShiftHelper::MatrixXf;
    using MatrixXi = GridShiftHelper::MatrixXi;

    const GridShiftHelper& grid;

    // Pre-allocated buffers for vectorized operations
    MatrixXf neighborSum;           // Sum of neighbor concentrations
    MatrixXf laplacian;             // Discrete Laplacian operator result
    MatrixXf soilMask;              // Precomputed mask for soil layer cells
    MatrixXf cellMask;              // Mask for living cells
    MatrixXf absorbed;              // Amount absorbed by each cell
    
    void precomputeSoilMask(int soilLayerHeight);
};
