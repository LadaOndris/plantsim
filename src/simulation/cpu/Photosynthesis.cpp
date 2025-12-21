#include "simulation/cpu/Photosynthesis.h"

void Photosynthesis::apply(State& state, const Options& options) {
    // Calculate storage dimensions from logical dimensions
    // Storage width is larger due to hexagonal stagger pattern
    const int h = state.height;
    const int w = state.width + (state.height - 1) / 2; // TODO: use GridTopology method?
    
    Eigen::Map<const MatrixXi> cellTypes(state.cellTypes.data(), h, w);
    Eigen::Map<const MatrixXf> light(state.light.data(), h, w);
    Eigen::Map<const MatrixXf> water(state.plantWater.data(), h, w);
    Eigen::Map<MatrixXf> sugar(state.plantSugar.data(), h, w);
    
    auto isPlant = (cellTypes.array() == static_cast<int>(CellState::Type::Cell)).cast<float>();
    
    // Compute saturation terms using Michaelis-Menten kinetics
    // light_term = light / (light + k)
    auto lightTerm = light.array() / (light.array() + options.lightHalfSat + 1e-12f);
    auto waterTerm = water.array() / (water.array() + options.waterHalfSat + 1e-12f);
    
    auto sugarProduced = options.dt * options.photoMaxRate * lightTerm * waterTerm * isPlant;
    
    sugar.array() += sugarProduced;
}
