#include "simulation/cpu/NutrientDiffusion.h"
#include "simulation/initializers/Initializers.h"
#include "simulation/initializers/amounts/FixedAmount.h"
#include <Eigen/Dense>

NutrientDiffusion::NutrientDiffusion(const GridShiftHelper& grid, const Options& options)
    : grid(grid)
{
    const int h = grid.height();
    const int w = grid.width();
    
    neighborSum.resize(h, w);
    laplacian.resize(h, w);
    soilMask.resize(h, w);
    cellMask.resize(h, w);
    absorbed.resize(h, w);

    precomputeSoilMask(options.soilLayerHeight);
}

void NutrientDiffusion::precomputeSoilMask(int soilLayerHeight) {
    using namespace initializers;
    
    const int h = grid.height();
    const int w = grid.width();
    const auto& validity = grid.getValidityMask();
    
    GridTopology topology{w, h};
    
    soilMask.setZero();
    
    PolicyApplication soilPolicy{BottomRowsRegion{soilLayerHeight}, SetValue{FixedAmount{1.0f}}};
    soilPolicy.apply(topology, soilMask);
}

void NutrientDiffusion::step(State& state, State& backBuffer, const Options& options) {
    if (!options.enableNutrients) {
        return;
    }

    const int h = grid.height();
    const int w = grid.width();
    const auto& validity = grid.getValidityMask();

    Eigen::Map<const MatrixXf> nutrients(state.nutrients.data(), h, w);
    Eigen::Map<MatrixXf> nextNutrients(backBuffer.nutrients.data(), h, w);
    Eigen::Map<const MatrixXi> cellTypes(state.cellTypes.data(), h, w);
    Eigen::Map<const MatrixXf> resources(state.resources.data(), h, w);
    Eigen::Map<MatrixXf> nextResources(backBuffer.resources.data(), h, w);

    // ========== Phase 1: Diffusion ==========
    // Compute discrete Laplacian: sum(neighbors) - 6*center
    // This drives diffusion from high to low concentration
    neighborSum.setZero();
    for (const auto& shift : grid.getIncomingShifts()) {
        grid.accumulateShifted(neighborSum, nutrients, shift);
    }
    
    // Laplacian = sum(neighbors) - 6*center
    laplacian = (neighborSum - 6.0f * nutrients).cwiseProduct(soilMask);
    
    // Apply diffusion: nutrients += diffusionRate * laplacian
    nextNutrients.noalias() = nutrients + options.nutrientDiffusionRate * laplacian;

    // ========== Phase 2: Absorption by cells ==========
    // Living cells absorb nutrients from their location
    cellMask = (validity.array() * 
        (cellTypes.array() == static_cast<int>(CellState::Type::Cell)).cast<float>()).matrix();
    
    // Each cell absorbs min(absorptionRate, available nutrients)
    absorbed = cellMask.cwiseProduct(
        nextNutrients.cwiseMin(options.nutrientAbsorptionRate)
    );
    
    // Remove absorbed nutrients from the field
    nextNutrients -= absorbed;
    
    // Add absorbed nutrients to cell resources
    nextResources.noalias() = resources + absorbed;

    // ========== Phase 3: Regeneration in soil layer ==========
    // Regenerate: nutrients += regenerationRate * (maxNutrient - nutrients) * soilMask
    nextNutrients += soilMask.cwiseProduct(
        (options.nutrientRegenerationRate * (options.maxNutrient - nextNutrients.array())).matrix()
    );

    // ========== Phase 4: Clamping ==========
    // Clamp nutrients to [0, maxNutrient]
    nextNutrients = nextNutrients.cwiseMax(0.0f).cwiseMin(options.maxNutrient);

    std::swap(state.nutrients, backBuffer.nutrients);
    std::swap(state.resources, backBuffer.resources);
}
