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
    auto [width, height] = map.getStorageDims();

    // Extract resource and type data from map into simple arrays
    std::vector<int> resources(width * height);
    std::vector<int> types(width * height);

    for (int r = 0; r < height; ++r) {
        for (int q = 0; q < width; ++q) {
            const auto &cell = map.getCellAt(r, q);
            resources[r * width + q] = cell.resources;
            types[r * width + q] = static_cast<int>(cell.type);
        }
    }

    // Prepare a delta buffer for moved resources
    std::vector<int> delta(width * height, 0);

    // Set up SYCL queue (use default device, could be GPU)
    queue q{default_selector_v};

    // Create SYCL buffers on device
    buffer<int, 1> resBuf(resources.data(), range<1>(resources.size()));
    buffer<int, 1> typeBuf(types.data(), range<1>(types.size()));
    buffer<int, 1> deltaBuf(delta.data(), range<1>(delta.size()));
    buffer<std::pair<int,int>, 1> offsetBuf(neighborOffsets.data(),
                                            range<1>(neighborOffsets.size()));

    q.submit([&](handler &h) {
        // Accessors
        auto res = resBuf.get_access<access::mode::read>(h);
        auto type = typeBuf.get_access<access::mode::read>(h);
        auto deltaAcc = deltaBuf.get_access<access::mode::atomic>(h);
        auto offs = offsetBuf.get_access<access::mode::read>(h);

        h.parallel_for(range<2>(height, width), [=](id<2> idx) {
            int r = idx[0];
            int qx = idx[1];
            int idxFlat = r * width + qx;

            if (type[idxFlat] != CellState::Type::Cell) return;

            int selfRes = res[idxFlat];
            if (selfRes <= 0) return;

            for (int i = 0; i < offs.size(); ++i) {
                int dq = offs[i].first;
                int dr = offs[i].second;
                int nr = r + dr;
                int nq = qx + dq;

                if (nr < 0 || nq < 0 || nr >= height || nq >= width)
                    continue;

                int nIdx = nr * width + nq;

                if (type[nIdx] != CellState::Type::Cell)
                    continue;

                // Each move transfers one resource (could adapt to your logic)
                int moveResource = 1;

                // Atomically record deltas
                deltaAcc[idxFlat].fetch_sub(moveResource);
                deltaAcc[nIdx].fetch_add(moveResource);
            }
        });
    });

    q.wait();

    // Retrieve delta results back to host
    host_accessor deltaHost(deltaBuf, read_only);

    // Second sweep: apply delta to map
    for (int r = 0; r < height; ++r) {
        for (int qx = 0; qx < width; ++qx) {
            int idxFlat = r * width + qx;
            map.getCellAt(r, qx).resources += deltaHost[idxFlat];
        }
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