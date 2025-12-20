#include "simulation/cpu/SoilDiffusion.h"
#include "simulation/initializers/Initializers.h"
#include "simulation/initializers/amounts/FixedAmount.h"
#include <Eigen/Dense>

SoilDiffusion::SoilDiffusion(const GridShiftHelper& grid, const Options& options)
    : grid(grid)
{
    const int h = grid.height();
    const int w = grid.width();
    
    neighborSum.resize(h, w);
    neighborCount.resize(h, w);
    avgNeighbors.resize(h, w);
    soilMask.resize(h, w);
    tempBuffer.resize(h, w);

    precomputeSoilMask(options.soilLayerHeight);
}

void SoilDiffusion::precomputeSoilMask(int soilLayerHeight) {
    using namespace initializers;
    
    const int h = grid.height();
    const int w = grid.width();
    const auto& validity = grid.getValidityMask();
    
    GridTopology topology{w, h};
    
    soilMask.setZero();
    
    PolicyApplication soilPolicy{BottomRowsRegion{soilLayerHeight}, SetValue{FixedAmount{1.0f}}};
    soilPolicy.apply(topology, soilMask);
}

void SoilDiffusion::step(State& state, State& backBuffer, const Options& options) {
    if (!options.enableSoilSystem) {
        return;
    }

    const int h = grid.height();
    const int w = grid.width();
    const float dt = options.dt;

    // Map state vectors to Eigen matrices
    Eigen::Map<MatrixXf> water(state.soilWater.data(), h, w);
    Eigen::Map<MatrixXf> mineral(state.soilMineral.data(), h, w);
    Eigen::Map<MatrixXf> nextWater(backBuffer.soilWater.data(), h, w);
    Eigen::Map<MatrixXf> nextMineral(backBuffer.soilMineral.data(), h, w);

    // Copy current state to back buffer as working copy
    nextWater = water;
    nextMineral = mineral;

    // ========== Phase 1: Regeneration ==========
    // Soil regenerates toward target values
    applyRegeneration(nextWater, options.soilWaterTarget, options.soilWaterRegenRate, dt);
    applyRegeneration(nextMineral, options.soilMineralTarget, options.soilMineralRegenRate, dt);

    // ========== Phase 2: Diffusion ==========
    // Resources diffuse between neighboring soil tiles
    applyDiffusion(nextWater, options.soilWaterDiffusivity, dt);
    applyDiffusion(nextMineral, options.soilMineralDiffusivity, dt);

    // ========== Phase 3: Clamping ==========
    // Ensure non-negative values
    nextWater = nextWater.cwiseMax(0.0f);
    nextMineral = nextMineral.cwiseMax(0.0f);

    // Swap buffers
    std::swap(state.soilWater, backBuffer.soilWater);
    std::swap(state.soilMineral, backBuffer.soilMineral);
}

void SoilDiffusion::applyRegeneration(MatrixXf& field, float target, float regenRate, float dt) {
    // Regeneration formula: field += dt * regenRate * (target - field) * soilMask
    // This pulls field values toward target in soil regions only
    field += soilMask.cwiseProduct(
        (dt * regenRate * (target - field.array())).matrix()
    );
}

void SoilDiffusion::applyDiffusion(MatrixXf& field, float diffusivity, float dt) {
    // Compute average of soil neighbors for hex grid diffusion
    // Only soil tiles contribute to and receive diffusion
    
    neighborSum.setZero();
    neighborCount.setZero();
    
    // State-dependent: only consider soil tiles as valid neighbors
    MatrixXf soilField = field.cwiseProduct(soilMask);
    
    // Accumulate neighbor values and counts from all 6 hex directions
    for (const auto& shift : grid.getIncomingShifts()) {
        grid.accumulateShifted(neighborSum, soilField, shift);
        grid.accumulateShifted(neighborCount, soilMask, shift);
    }
    
    // Compute average (avoid division by zero)
    avgNeighbors = (neighborCount.array() > 0.0f)
        .select(neighborSum.array() / neighborCount.array(), field.array());
    
    // Apply diffusion: field += dt * diffusivity * (avgNeighbors - field) * soilMask
    // Only soil tiles are updated
    field += soilMask.cwiseProduct(
        (dt * diffusivity * (avgNeighbors.array() - field.array())).matrix()
    );
}
