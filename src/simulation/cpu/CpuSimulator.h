#pragma once

#include "simulation/GridTopology.h"
#include "simulation/State.h"
#include "simulation/Options.h"
#include "simulation/ISimulator.h"
#include "simulation/CellState.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cstring>

struct ShiftInfo {
    int dstRow;
    int dstCol;
    int srcRow;
    int srcCol;
    int copyH;
    int copyW;
};

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
    std::vector<ShiftInfo> outgoingShifts;
    std::vector<ShiftInfo> incomingShifts;
    
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

        const size_t N = topology.neighborOffsets.size();
        outgoingShifts.resize(N);
        incomingShifts.resize(N);

        for (size_t i = 0; i < N; ++i) {
            const auto& off = topology.neighborOffsets[i];

            // OUTGOING uses -offset
            outgoingShifts[i] = computeShiftInfo(-off.r, -off.q, storageHeight, storageWidth);

            // INCOMING uses +offset
            incomingShifts[i] = computeShiftInfo(+off.r, +off.q, storageHeight, storageWidth);
        }
    }

    inline ShiftInfo computeShiftInfo(int dy, int dx, int H, int W)
    {
        ShiftInfo info;

        info.dstRow = (dy >= 0 ? dy : 0);
        info.dstCol = (dx >= 0 ? dx : 0);

        info.srcRow = (dy >= 0 ? 0 : -dy);
        info.srcCol = (dx >= 0 ? 0 : -dx);

        info.copyH = H - std::abs(dy);
        info.copyW = W - std::abs(dx);

        bool isValid = (info.copyH > 0 && info.copyW > 0);
        if (!isValid) {
            throw std::runtime_error("Invalid shift computed in precomputeTopology: zero-sized block.");
        }

        return info;
    }

    template <typename Deriveddst, typename Derivedsrc>
    void accumulateShifted(Eigen::MatrixBase<Deriveddst>& dst,
                        const Eigen::MatrixBase<Derivedsrc>& src,
                        const ShiftInfo& s) {
        dst.block(s.dstRow, s.dstCol, s.copyH, s.copyW).noalias() +=
            src.block(s.srcRow, s.srcCol, s.copyH, s.copyW);
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
        for (const auto& shift : outgoingShifts) {
            accumulateShifted(neighborsCanReceiveCount, receiverMask, shift);
        }

        auto availableOutflow = (resources.array() * receiverMask.array()).matrix();
        auto totalOutgoing = availableOutflow.cwiseMin(neighborsCanReceiveCount);
        flowPerNeighbor.array() = (neighborsCanReceiveCount.array() != 0)
                .select(totalOutgoing.array() / neighborsCanReceiveCount.array(), 0);

        totalIncoming.setZero();

        for (const auto& shift : incomingShifts) {
            accumulateShifted(totalIncoming, flowPerNeighbor, shift);
        }
        totalIncoming = (totalIncoming.array() * receiverMask.array()).matrix();

        newResources.noalias() = resources - totalOutgoing + totalIncoming;

        std::memcpy(state.resources.data(), newResources.data(), 
                    storageWidth * storageHeight * sizeof(int));
    }
};
