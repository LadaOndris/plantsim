#pragma once

#include "simulation/State.h"
#include "simulation/Options.h"
#include "simulation/CellState.h"
#include <Eigen/Dense>

/**
 * @brief Computes photosynthesis for plant cells.
 * 
 * Photosynthesis converts light energy and water into sugar (energy storage).
 * Production is limited by both light availability and internal water stores.
 */
class Photosynthesis {
public:
    using MatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using MatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    /**
     * @brief Apply photosynthesis to all plant cells.
     * 
     * Increases plantSugar based on available light and water using
     * Michaelis-Menten style saturation curves.
     * 
     * @param state Current simulation state (plantSugar will be updated)
     * @param options Simulation parameters (rates, half-saturation constants, dt)
     */
    static void apply(State& state, const Options& options);
};
