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
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> validNeighborCounts;
    std::vector<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> neighborValidityCache;

    /**
     * @brief Precomputes topology information since grid structure is static.
     * 
     * This computes:
     * - validityMask: which cells are valid (not padding)
     * - validNeighborCounts: how many valid neighbors each cell has
     * - neighborValidityCache: for each direction, which cells have a valid neighbor in that direction
     */
    void precomputeTopology() {
        const StorageCoord storageDims = topology.getStorageDimension();
        const int storageWidth = storageDims.x;
        const int storageHeight = storageDims.y;

        validityMask.resize(storageHeight, storageWidth);
        validityMask.setZero();
        validNeighborCounts.resize(storageHeight, storageWidth);
        validNeighborCounts.setZero();
        neighborValidityCache.resize(topology.neighborOffsets.size());

        // Build validity mask
        for (int y = 0; y < storageHeight; y++) {
            for (int x = 0; x < storageWidth; x++) {
                validityMask(y, x) = topology.isValid(StorageCoord{x, y}) ? 1 : 0;
            }
        }

        // For each neighbor direction, compute shifted validity and accumulate counts
        for (size_t dirIdx = 0; dirIdx < topology.neighborOffsets.size(); dirIdx++) {
            const auto& offset = topology.neighborOffsets[dirIdx];
            const int dq = offset.q;
            const int dr = offset.r;

            neighborValidityCache[dirIdx].resize(storageHeight, storageWidth);
            neighborValidityCache[dirIdx].setZero();

            // Shift validity mask by (-dq, -dr) to get neighbor validity at each cell
            shiftMatrix(validityMask.data(), neighborValidityCache[dirIdx].data(),
                        -dr, -dq, storageWidth, storageHeight);

            validNeighborCounts += neighborValidityCache[dirIdx];
        }
    }

    /**
     * @brief Shifts a matrix by the given offset, filling shifted-in values with 0.
     * 
     * @param src Source data
     * @param dst Destination data
     * @param dy Row offset (positive = shift down)
     * @param dx Column offset (positive = shift right)
     * @param width Storage width
     * @param height Storage height
     */
    static void shiftMatrix(const int* __restrict src, int* __restrict dst,
                            int dy, int dx, int width, int height) {
        Eigen::Map<const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
            srcMat(src, height, width);
        Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
            dstMat(dst, height, width);
        
        dstMat.setZero();
        
        const int srcRowStart = std::max(0, -dy);
        const int srcRowEnd = std::min(height, height - dy);
        const int srcColStart = std::max(0, -dx);
        const int srcColEnd = std::min(width, width - dx);
        
        const int dstRowStart = std::max(0, dy);
        const int dstColStart = std::max(0, dx);
        
        const int copyRows = srcRowEnd - srcRowStart;
        const int copyCols = srcColEnd - srcColStart;
        
        if (copyRows > 0 && copyCols > 0) {
            dstMat.block(dstRowStart, dstColStart, copyRows, copyCols) = 
                srcMat.block(srcRowStart, srcColStart, copyRows, copyCols);
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

        // Map state vectors to Eigen matrices
        Eigen::Map<const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
            resources(state.resources.data(), storageHeight, storageWidth);
        Eigen::Map<const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
            cellTypes(state.cellTypes.data(), storageHeight, storageWidth);

        // receiver_mask = isValidCell & (cellTypes == CellType::Cell)
        Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> receiverMask =
            (validityMask.array() * (cellTypes.array() == static_cast<int>(CellState::Type::Cell)).cast<int>()).matrix();

        // neighborsCanReceiveCount = sum of shifted receiver masks
        Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> neighborsCanReceiveCount(storageHeight, storageWidth);
        neighborsCanReceiveCount.setZero();
        
        std::vector<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> neighborReceiverMask(topology.neighborOffsets.size());
        
        for (size_t dirIdx = 0; dirIdx < topology.neighborOffsets.size(); dirIdx++) {
            const auto& offset = topology.neighborOffsets[dirIdx];
            const int dq = offset.q;
            const int dr = offset.r;

            neighborReceiverMask[dirIdx].resize(storageHeight, storageWidth);
            neighborReceiverMask[dirIdx].setZero();

            // Shift receiver_mask by (-direction) to move the neighbor on top of current cell
            shiftMatrix(receiverMask.data(), neighborReceiverMask[dirIdx].data(),
                        -dr, -dq, storageWidth, storageHeight);

            neighborsCanReceiveCount += neighborReceiverMask[dirIdx];
        }

        // outflowPerNeighbor = 1
        constexpr int outflowPerNeighbor = 1;
        
        // desiredOutflow = neighborsCanReceiveCount * outflowPerNeighbor
        Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> desiredOutflow = 
            neighborsCanReceiveCount * outflowPerNeighbor;

        // sender_mask based on receiver_mask (hasResources check is implicit in final multiplication)
        Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> senderMask = receiverMask;

        // availableOutflow = resources * sender_mask
        // If resources[i][j] == 0, result is 0 regardless of senderMask
        Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> availableOutflow =
            (resources.array() * senderMask.array()).matrix();

        // totalOutgoing = min(availableOutflow, desiredOutflow)
        Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> totalOutgoing =
            availableOutflow.cwiseMin(desiredOutflow);

        // flowPerNeighbor = totalOutgoing / neighborsCanReceiveCount (where neighborsCanReceiveCount != 0, otherwise 0)
        Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> flowPerNeighbor(storageHeight, storageWidth);
        flowPerNeighbor =
            (neighborsCanReceiveCount.array() != 0)
                .select(
                    totalOutgoing.array() / neighborsCanReceiveCount.array(),
                    0
                )
                .matrix();

        // totalIncoming = sum of shifted (flowPerNeighbor * neighbor_is_receiver)
        Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> totalIncoming(storageHeight, storageWidth);
        totalIncoming.setZero();
        
        Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> shiftedFlow(storageHeight, storageWidth);
        
        for (size_t dirIdx = 0; dirIdx < topology.neighborOffsets.size(); dirIdx++) {
            const auto& offset = topology.neighborOffsets[dirIdx];
            const int dq = offset.q;
            const int dr = offset.r;

            // flowInDirection = flowPerNeighbor * neighbor_is_receiver
            Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> flowInDirection =
                (flowPerNeighbor.array() * neighborReceiverMask[dirIdx].array()).matrix();

            // Shift flowInDirection by (+direction) to move the flow to the neighbor position
            shiftedFlow.setZero();
            shiftMatrix(flowInDirection.data(), shiftedFlow.data(),
                        dr, dq, storageWidth, storageHeight);

            totalIncoming += shiftedFlow;
        }

        // newResources = resources - totalOutgoing + totalIncoming
        Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> newResources =
            resources - totalOutgoing + totalIncoming;

        // Copy result to state
        std::memcpy(state.resources.data(), newResources.data(), 
                    storageWidth * storageHeight * sizeof(int));
    }
};
