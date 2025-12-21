#pragma once

#include "simulation/cpu/GridShiftHelper.h"
#include "simulation/State.h"
#include "simulation/Options.h"
#include "simulation/CellState.h"

/**
 * @brief Internal transport of resources within plant cells using diffusion.
 * 
 * Implements the Python reference algorithm:
 *   X += dt * T * (avg_plant_neighbors(X) - X)
 * 
 * Only PLANT cells participate. Each resource (sugar, water, mineral) has its
 * own transport rate.
 */
class ResourceTransfer {
public:
    explicit ResourceTransfer(const GridShiftHelper& grid);

    void step(State& state, State& backBuffer, const Options& options);

private:
    using MatrixXf = GridShiftHelper::MatrixXf;
    using MatrixXi = GridShiftHelper::MatrixXi;

    const GridShiftHelper& grid;

    /**
     * @brief Apply diffusion-based transport to a single resource field.
     * 
     * @param resource Current resource values (h x w)
     * @param nextResource Output buffer for updated values
     * @param plantMask Mask indicating which cells are plants
     * @param transportRate Diffusion rate for this resource
     * @param dt Time step
     */
    void applyTransport(
        const MatrixXf& resource,
        Eigen::Ref<MatrixXf> nextResource,
        const MatrixXf& plantMask,
        float transportRate,
        float dt
    );

    // Pre-allocated buffers
    MatrixXf plantMask;
    MatrixXf neighborSum;
    MatrixXf neighborCount;
    MatrixXf avgNeighbor;
};
