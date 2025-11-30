#pragma once

#include "simulation/GridTopology.h"
#include "simulation/State.h"
#include "simulation/Options.h"
#include "simulation/ISimulator.h"
#include "simulation/CellState.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cstring>

/**
 * @brief CPU-based simulator implementation using matrix operations.
 * 
 * This implementation follows a vectorized approach similar to NumPy-based
 * simulation, precomputing topology and using matrix shifts for neighbor
 * operations.
 */
class CpuSimulator : public ISimulator {
public:
    explicit CpuSimulator(State initialState) 
        : state(std::move(initialState))
        , backBuffer(state)
        , topology(state.width, state.height)
    {
        precomputeTopology();
    }
    
    const State &getState() const override {
        return state;
    }

    void step(const Options &options) override {
        if (options.enableResourceTransfer) {
            transferResources();
        }
    }

private:
    State state;
    State backBuffer;
    GridTopology topology;
    
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> validityMask;

    // Pre-allocated buffers to avoid memory allocation in the simulation loop
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> receiverMask;
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> neighborsCanReceiveCount;
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> flowPerNeighbor;
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> totalIncoming;
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> newResources;

    /**
     * @brief Precomputes topology information since grid structure is static.
     */
    void precomputeTopology() {
        const StorageCoord storageDims = topology.getStorageDimension();
        const int storageWidth = storageDims.x;
        const int storageHeight = storageDims.y;

        validityMask.resize(storageHeight, storageWidth);
        validityMask.setZero();

        // Resize persistent buffers to avoid reallocation in step()
        receiverMask.resize(storageHeight, storageWidth);
        neighborsCanReceiveCount.resize(storageHeight, storageWidth);
        flowPerNeighbor.resize(storageHeight, storageWidth);
        totalIncoming.resize(storageHeight, storageWidth);
        newResources.resize(storageHeight, storageWidth);

        // Build validity mask
        for (int y = 0; y < storageHeight; y++) {
            for (int x = 0; x < storageWidth; x++) {
                validityMask(y, x) = topology.isValid(StorageCoord{x, y}) ? 1 : 0;
            }
        }
    }

    template <typename Deriveddst, typename Derivedsrc>
    void accumulateShifted(Eigen::MatrixBase<Deriveddst>& dst,
                        const Eigen::MatrixBase<Derivedsrc>& src,
                        int dy, int dx) {
        const int h = dst.rows();
        const int w = dst.cols();

        // Calculate the intersection (the valid block to copy)
        // Positive dy means we shift "Down", so we write to dst starting at dy
        const int dstRow = std::max(0, dy);
        const int dstCol = std::max(0, dx);
        const int srcRow = std::max(0, -dy);
        const int srcCol = std::max(0, -dx);

        const int copyH = h - std::abs(dy);
        const int copyW = w - std::abs(dx);

        if (copyH > 0 && copyW > 0) {
            // DIRECT VIEW OPERATION: No malloc, no memcpy. 
            // Just adding one sub-block to another.
            dst.block(dstRow, dstCol, copyH, copyW) += 
                src.block(srcRow, srcCol, copyH, copyW);
        }
    }

    /**
     * @brief Performs one step of resource redistribution using matrix operations.
     * 
     * Algorithm (matching Python implementation):
     * 1. Identify active sources (cells with resources > 0 and type == Cell)
     * 2. Calculate outgoing flow: min(resources, validNeighborCount) for active cells
     * 3. Calculate flow per neighbor direction
     * 4. Shift flow matrices to compute incoming flow
     * 5. Update: new_resources = resources - outgoing + incoming
     */
    void transferResources() {
        const StorageCoord storageDims = topology.getStorageDimension();
        const int storageWidth = storageDims.x;
        const int storageHeight = storageDims.y;

        Eigen::Map<const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
            resources(state.resources.data(), storageHeight, storageWidth);
        Eigen::Map<const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
            cellTypes(state.cellTypes.data(), storageHeight, storageWidth);

        receiverMask = (validityMask.array() * (cellTypes.array() == static_cast<int>(CellState::Type::Cell)).cast<int>()).matrix();

        neighborsCanReceiveCount.setZero();
        for (const auto& offset : topology.neighborOffsets) {
            accumulateShifted(neighborsCanReceiveCount, receiverMask, -offset.r, -offset.q);
        }

        auto availableOutflow = (resources.array() * receiverMask.array()).matrix();
        auto totalOutgoing = availableOutflow.cwiseMin(neighborsCanReceiveCount);
        flowPerNeighbor.array() = (neighborsCanReceiveCount.array() != 0)
                .select(totalOutgoing.array() / neighborsCanReceiveCount.array(), 0);

        totalIncoming.setZero();

        for (const auto& offset : topology.neighborOffsets) {
            accumulateShifted(totalIncoming, flowPerNeighbor, offset.r, offset.q);
        }
        totalIncoming = (totalIncoming.array() * receiverMask.array()).matrix();

        newResources.noalias() = resources - totalOutgoing + totalIncoming;

        std::memcpy(state.resources.data(), newResources.data(), 
                    storageWidth * storageHeight * sizeof(int));
    }
};
