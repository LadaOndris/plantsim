#include "simulation/cpu/ResourceTransfer.h"

ResourceTransfer::ResourceTransfer(const GridShiftHelper& grid)
    : grid(grid)
{
    const int h = grid.height();
    const int w = grid.width();
    
    receiverMask.resize(h, w);
    neighborsCanReceiveCount.resize(h, w);
    flowPerNeighbor.resize(h, w);
    totalIncoming.resize(h, w);
}

void ResourceTransfer::step(State& state, State& backBuffer, const Options& options) {
    if (!options.enableResourceTransfer) {
        return;
    }

    const int h = grid.height();
    const int w = grid.width();
    const auto& validity = grid.getValidityMask();

    Eigen::Map<const MatrixXf> resources(state.resources.data(), h, w);
    Eigen::Map<MatrixXf> nextResources(backBuffer.resources.data(), h, w);
    Eigen::Map<const MatrixXi> cellTypes(state.cellTypes.data(), h, w);

    // Cells that can receive resources
    receiverMask = (validity.array() * 
        (cellTypes.array() == static_cast<int>(CellState::Type::Cell)).cast<float>()).matrix();

    // Count how many neighbors can receive
    neighborsCanReceiveCount.setZero();
    for (const auto& shift : grid.getOutgoingShifts()) {
        grid.accumulateShifted(neighborsCanReceiveCount, receiverMask, shift);
    }

    // Calculate flow
    auto availableOutflow = (resources.array() * receiverMask.array()).matrix();
    auto totalOutgoing = availableOutflow.cwiseMin(neighborsCanReceiveCount);
    flowPerNeighbor.array() = (neighborsCanReceiveCount.array() != 0)
            .select(totalOutgoing.array() / neighborsCanReceiveCount.array(), 0);

    // Accumulate incoming flow
    totalIncoming.setZero();
    for (const auto& shift : grid.getIncomingShifts()) {
        grid.accumulateShifted(totalIncoming, flowPerNeighbor, shift);
    }
    totalIncoming = (totalIncoming.array() * receiverMask.array()).matrix();

    // Update resources
    nextResources.noalias() = resources - totalOutgoing + totalIncoming;
    std::swap(state.resources, backBuffer.resources);
}
