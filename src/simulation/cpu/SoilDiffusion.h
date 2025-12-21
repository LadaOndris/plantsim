#pragma once

#include "simulation/cpu/GridShiftHelper.h"
#include "simulation/State.h"
#include "simulation/Options.h"
#include "simulation/CellState.h"

/**
 * @brief Soil resource (water & mineral) diffusion and regeneration system.
 * 
 * Manages diffusive water and mineral fields on the hex grid with:
 * - State-dependent diffusion: only SOIL tiles diffuse resources
 * - Regeneration: soil layer regenerates toward target values
 * - Separate diffusivity rates for water vs minerals
 * - Regeneration: soil += dt * regen_rate * (target - soil)
 * - Diffusion: X += dt * D * (avg_neighbors - X) only where state == SOIL
 */
class SoilDiffusion {
public:
    explicit SoilDiffusion(const GridShiftHelper& grid, const Options& options);

    /**
     * @brief Perform one step of soil resource update.
     * 
     * Executes two phases for each resource (water and mineral):
     * 1. Regeneration: Soil tiles regenerate toward target values
     * 2. Diffusion: Resources diffuse between neighboring soil tiles
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
    MatrixXf neighborCount;         // Count of valid soil neighbors
    MatrixXf avgNeighbors;          // Average of neighbor values
    MatrixXf soilMask;              // Precomputed mask for soil layer cells
    MatrixXf tempBuffer;            // Temporary buffer for computations
    
    void precomputeSoilMask(int soilLayerHeight);
    
    /**
     * @brief Apply regeneration to a resource field in soil regions.
     * 
     * Formula: field += dt * regenRate * (target - field) * soilMask
     * 
     * @param field The resource field to regenerate (in-place modification)
     * @param target The target equilibrium value
     * @param regenRate The regeneration rate
     * @param dt Time step
     */
    void applyRegeneration(Eigen::Ref<MatrixXf> field, float target, float regenRate, float dt);
    
    /**
     * @brief Apply diffusion to a resource field between soil tiles.
     * 
     * Formula: field += dt * diffusivity * (avgNeighbors - field) * soilMask
     * Only soil tiles participate in diffusion.
     * 
     * @param field The resource field to diffuse (in-place modification)
     * @param diffusivity The diffusion rate
     * @param dt Time step
     */
    void applyDiffusion(Eigen::Ref<MatrixXf> field, float diffusivity, float dt);
};
