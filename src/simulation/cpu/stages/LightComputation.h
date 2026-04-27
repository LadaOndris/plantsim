#pragma once

#include "simulation/State.h"
#include "simulation/Options.h"
#include "simulation/CellState.h"
#include <Eigen/Dense>

/**
 * @brief Computes directional light propagation from top to bottom.
 */
class LightComputation {
public:
    using MatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    /**
     * @brief Compute light field for the entire grid.
     * 
     * Updates state.light with incident light values.
     * Light propagates top-down, being attenuated by each cell type.
     * 
     * @param state Current simulation state (light field will be updated)
     * @param options Simulation parameters (absorption rates, top intensity)
     */
    static void compute(State& state, const Options& options);
};
