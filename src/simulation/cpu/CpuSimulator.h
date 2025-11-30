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
        const int totalCells = storageWidth * storageHeight;

        const int* srcResources = state.resources.data();
        const int* srcCellTypes = state.cellTypes.data();
        
        // isValidCell - precomputed as a matrix
        // cellTypes - stored as a matrix from external data

        // receiver_mask = isValidCell & (cellTypes == CellType::Cell)
        // neighborsCanReceiveCount = np.zeros_like(isValidCell, type=int)
        // for i, direction in enumerate(neighbor_directions):
        //    neighbor_receiver_mask[i] = shiftMatrix(receiver_mask, -direction) // Move the neighbor on top of current cell
        //    neighborsCanReceiveCount += neighbor_receiver_mask[i]

        // outflowPerNeighbor = 1
        // desiredOutflow = neighborsCanReceiveCount * outflowPerNeighbor

        // hasResources - computed here as resources > 0
        // sender_mask = receiver_mask * hasResources
        // availableOutflow = resources * sender_mask
        // totalOutgoing = min(availableOutflow, desiredOutflow)

        // flowPerNeighbor = totalOutgoing / neighborsCanReceiveCount (where neighborsCanReceiveCount != 0, otherwise 0)

        // for i, direction in enumerate(neighbor_directions):
        //    neighbor_is_receiver = neighbor_receiver_mask[i]
        //    flowInDirection = flowPerNeighbor * neighbor_is_receiver
        //    totalIncoming += shiftMatrix(flowInDirection, direction) // Move the flow to the neighbor position

        // newResources = resources - totalOutgoing + totalIncoming


        // Copy result to state
        //std::memcpy(state.resources.data(), newResources.data(), totalCells * sizeof(float));
    }
};
