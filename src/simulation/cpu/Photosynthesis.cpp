#include "simulation/cpu/Photosynthesis.h"

void Photosynthesis::apply(State& state, const Options& options) {
    // Calculate storage dimensions from logical dimensions
    const int h = state.height;
    const int w = state.width + (state.height - 1) / 2; // TODO: use GridTopology method?

    Eigen::Map<const MatrixXi> cellTypes(state.cellTypes.data(), h, w);
    Eigen::Map<const MatrixXf> light(state.light.data(), h, w);

    Eigen::Map<MatrixXf> water(state.plantWater.data(), h, w);
    Eigen::Map<MatrixXf> sugar(state.plantSugar.data(), h, w);

    const auto isPlant = (cellTypes.array() == static_cast<int>(CellState::Type::Cell)).cast<float>();

    // Michaelis–Menten saturation terms
    const auto lightTerm = light.array() / (light.array() + options.lightHalfSat + 1e-12f);
    const auto waterTerm = water.array() / (water.array() + options.waterHalfSat + 1e-12f);

    // Potential production rate from light + internal water status
    auto potentialSugar = options.dt * options.photoMaxRate * lightTerm * waterTerm * isPlant;

    // Hard cap by how much water is actually available in the cell.
    // Add epsilon to avoid division by zero, though water>=0 should hold.
    const auto maxSugarFromWater = (water.array() / (options.waterPerSugar + 1e-12f)) * isPlant;

    // Final sugar made cannot exceed available water.
    const auto sugarProduced = potentialSugar.min(maxSugarFromWater);

    sugar.array() += sugarProduced;

    // Water is consumed in proportion to sugar produced.
    water.array() -= sugarProduced * options.waterPerSugar;
}
