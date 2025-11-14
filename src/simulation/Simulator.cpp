//
// Created by lada on 4/13/24.
//

#include "Simulator.h"
#include "simulation/CellState.h"

#include "sycl/sycl.hpp"

Simulator::Simulator(WorldState &worldState) : worldState{worldState} {
    auto &map = worldState.getMap();
    auto &cells = map.getCells();

    std::vector<std::pair<int, int>> coords{
            {20, 20},
    };
    for (auto coord: coords) {
        int storageCoord = map.getStorageCoord(coord.second, coord.first);
        cells[storageCoord] = CellState(CellState::Type::Cell, 200000);
    }
}


void Simulator::step(const SimulatorOptions &options) {
    transferResources();
    replicateCells();
}

void Simulator::transferResources() {
    using namespace sycl;

    auto &map = worldState.getMap();
    auto neighborOffsets = map.getNeighborOffsets();
    std::pair<int, int> storageDims = map.getStorageDims();
    const int width = storageDims.first;
    const int height = storageDims.second;
    const int numNeighbors = neighborOffsets.size();

    auto &cells = map.getCells(); // assume vector<CellState>
    const size_t totalCells = width * height;

    // Host data
    std::vector<int> resources(totalCells);
    for (size_t i = 0; i < totalCells; ++i)
        resources[i] = cells[i].resources;
    try {
        // --- SYCL setup ---
        static queue q{gpu_selector_v};

        auto *neighborOffsetsUSM = malloc_shared<std::pair<int,int>>(numNeighbors, q);
        int *resUSM = malloc_shared<int>(totalCells, q);
        int *deltaUSM = malloc_shared<int>(height * width * numNeighbors, q);

        for (size_t i = 0; i < numNeighbors; ++i)
            neighborOffsetsUSM[i] = neighborOffsets[i];
    
        for (size_t i = 0; i < totalCells; ++i)
            resUSM[i] = cells[i].resources;

        // --- Pass 1: compute outgoing deltas ---
        q.parallel_for(range<2>(height, width), [=](id<2> idx) {
            const int r = idx[0];
            const int q_ = idx[1];
            const int flatIdx = r * width + q_;

            // zero out all neighbor deltas
            for (int n = 0; n < numNeighbors; ++n)
                deltaUSM[flatIdx * numNeighbors + n] = 0;

            // determine whether the cell can send
            int selfRes = resUSM[flatIdx];
            int canSend = selfRes > 0; // 1 if cell has resource, 0 otherwise

            // choose neighbor deterministically
            int chosenNeighbor = (r + q_) % numNeighbors;
            auto off = neighborOffsetsUSM[chosenNeighbor];
            int nr = r + off.second;
            int nq = q_ + off.first;

            // compute validity mask
            int valid = (nr >= 0) & (nr < height) & (nq >= 0) & (nq < width);

            // combine masks
            int mask = canSend & valid;

            // write the delta using the mask
            deltaUSM[flatIdx * numNeighbors + chosenNeighbor] = mask;
        }).wait();

        // --- Pass 2: accumulate deltas into resources ---
        q.parallel_for(range<2>(height, width), [=](id<2> idx) {
            const int r = idx[0];
            const int q_ = idx[1];
            const int flatIdx = r * width + q_;

            int gain = 0;
            int loss = 0;

            // Accumulate incoming deltas branchlessly
            for (int n = 0; n < numNeighbors; ++n) {
                auto off = neighborOffsetsUSM[n];
                int sr = r - off.second;
                int sq = q_ - off.first;

                // Compute mask: 1 if in bounds, 0 otherwise
                int valid = (sr >= 0) & (sr < height) & (sq >= 0) & (sq < width);
                int srcIdx = (sr * width + sq) * numNeighbors + n;

                // Use mask to add only valid deltas
                gain += deltaUSM[srcIdx] * valid;
            }

            // Outgoing delta (sent to chosen neighbor)
            for (int n = 0; n < numNeighbors; ++n)
                loss += deltaUSM[flatIdx * numNeighbors + n];

            resUSM[flatIdx] += gain - loss;
        }).wait();

        // Copy results back to host
        for (size_t i = 0; i < totalCells; ++i)
            cells[i].resources = resUSM[i];

        // Free USM memory
        free(neighborOffsetsUSM, q);
        free(resUSM, q);
        free(deltaUSM, q);
    } catch (const sycl::exception &e) {
        std::cerr << "SYCL Exception: " << e.what() << "\n";
        std::cerr << "Error code: " << e.code().value() << "\n";
    }
}

void Simulator::replicateCells() {
    auto &map = worldState.getMap();

    auto neighborOffsets = map.getNeighborOffsets();

    std::pair<int, int> storageDims = map.getStorageDims();

    for (int r = 0; r < storageDims.second; r++) {
        for (int q = 0; q < storageDims.first; q++) {
            for (auto offset: neighborOffsets) {
                CellState &pointCell = map.getCellAt(r, q);
                CellState &neighborCell = map.getCellAt(r + offset.second, q + offset.first);

                int canReplicate = pointCell.resources > 0 &&
                                   pointCell.type == CellState::Type::Cell && 
                                   neighborCell.type == CellState::Type::Air &&
                                   map.isValid(r, q) &&
                                   map.isValid(r + offset.second, q + offset.first);

                pointCell.resources -= canReplicate;
                neighborCell.type = canReplicate ? CellState::Type::Cell : neighborCell.type;
            }
        }
    }
}