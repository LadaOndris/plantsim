#include "simulation/cpu/ResourceTransfer.h"

ResourceTransfer::ResourceTransfer(const GridShiftHelper& grid)
    : grid(grid)
{
    const int h = grid.height();
    const int w = grid.width();
    
    plantMask.resize(h, w);
    neighborSum.resize(h, w);
    neighborCount.resize(h, w);
    avgNeighbor.resize(h, w);
}

void ResourceTransfer::applyTransport(
    const MatrixXf& resource,
    Eigen::Ref<MatrixXf> nextResource,
    const MatrixXf& mask,
    float transportRate,
    float dt
) {
    // Sum of plant neighbor values
    neighborSum.setZero();
    for (const auto& shift : grid.getIncomingShifts()) {
        // Only include values from plant cells
        grid.accumulateShifted(neighborSum, resource.cwiseProduct(mask), shift);
    }

    // Count of plant neighbors
    neighborCount.setZero();
    for (const auto& shift : grid.getIncomingShifts()) {
        grid.accumulateShifted(neighborCount, mask, shift);
    }

    // Average of plant neighbors (avoid division by zero)
    avgNeighbor = (neighborCount.array() > 0)
        .select(neighborSum.array() / neighborCount.array(), resource.array());

    // Apply diffusion: X += dt * T * (avg - X), only for plant cells
    nextResource = resource.array() + 
        mask.array() * dt * transportRate * (avgNeighbor.array() - resource.array());
}

void ResourceTransfer::step(State& state, State& backBuffer, const Options& options) {
    if (!options.enableResourceTransfer) {
        return;
    }

    const int h = grid.height();
    const int w = grid.width();
    const auto& validity = grid.getValidityMask();

    Eigen::Map<const MatrixXi> cellTypes(state.cellTypes.data(), h, w);

    // Only plant cells participate in internal transport
    plantMask = (validity.array() * 
        (cellTypes.array() == static_cast<int>(CellState::Type::Cell)).cast<float>()).matrix();

    // Map state vectors to matrices
    Eigen::Map<const MatrixXf> sugar(state.plantSugar.data(), h, w);
    Eigen::Map<const MatrixXf> water(state.plantWater.data(), h, w);
    Eigen::Map<const MatrixXf> mineral(state.plantMineral.data(), h, w);
    
    Eigen::Map<MatrixXf> nextSugar(backBuffer.plantSugar.data(), h, w);
    Eigen::Map<MatrixXf> nextWater(backBuffer.plantWater.data(), h, w);
    Eigen::Map<MatrixXf> nextMineral(backBuffer.plantMineral.data(), h, w);

    // Apply diffusion transport to each resource
    applyTransport(sugar, nextSugar, plantMask, options.sugarTransportRate, options.dt);
    applyTransport(water, nextWater, plantMask, options.waterTransportRate, options.dt);
    applyTransport(mineral, nextMineral, plantMask, options.mineralTransportRate, options.dt);

    // Swap buffers
    std::swap(state.plantSugar, backBuffer.plantSugar);
    std::swap(state.plantWater, backBuffer.plantWater);
    std::swap(state.plantMineral, backBuffer.plantMineral);
}
