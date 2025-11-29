#include <vector>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cstdlib>

#include "simulation/cpu/CpuSimulator.h"
#include "simulation/Options.h"
#include "simulation/GridTopology.h"
#include "simulation/State.h"
#include "simulation/MapPrinter.h"

template <typename T>
uint64_t computeChecksum(const std::vector<T> &cells, StorageCoord storageDims) {
    uint64_t checksum = 0;
    for (int y = 0; y < storageDims.y; y++) {
        for (int x = 0; x < storageDims.x; x++) {
            StorageCoord coord{.x=x, .y=y};
            checksum = checksum * 31 + static_cast<int>(cells[coord.asFlat(storageDims)]);
        }
    }
    return checksum;
}

State createInitialState(const GridTopology &topology) {
    const int width = topology.width;
    const int height = topology.height;
    const size_t totalCells = width * height;

    std::vector<int> resources(totalCells, 0);
    std::vector<int> cellTypes(totalCells, 0); // Air

    // Set up source cell with resources
    const AxialCoord cell{.q=1, .r=1};
    const int sourceIdx = cell.asFlat(topology.getDimension());
    resources[sourceIdx] = 1;
    cellTypes[sourceIdx] = 1; // Cell type

    // Set up neighboring cell (right neighbor)
    const AxialCoord neighbor{.q=2, .r=1};
    const int neighborIdx = neighbor.asFlat(topology.getDimension());
    cellTypes[neighborIdx] = 1; // Cell type

    // State should contain the storage data.
    auto storedResources = store(resources, width, height, -1);
    auto storedCellTypes = store(cellTypes, width, height, -1);

    return State(width, height, storedResources, storedCellTypes);
}

int main(int argc, char* argv[]) {
    int simSteps = 1;
    int gridSize = 20;

    if (argc > 1) {
        simSteps = std::atoi(argv[1]);
        if (simSteps <= 0) {
            std::cerr << "Usage: " << argv[0] << " [steps] [gridSize]" << std::endl;
            return 1;
        }
    }
    if (argc > 2) {
        gridSize = std::atoi(argv[2]);
        if (gridSize <= 0) {
            std::cerr << "Usage: " << argv[0] << " [steps] [gridSize]" << std::endl;
            return 1;
        }
    }

    std::cout << "=== Simulation Benchmark ===" << std::endl;
    std::cout << "Steps: " << simSteps << std::endl;
    std::cout << "Grid size: " << gridSize << "x" << gridSize << std::endl;
    std::cout << std::endl;

    GridTopology topology{gridSize, gridSize};
    State initialState = createInitialState(topology);
    std::cout << MapPrinter::printHexMapResources(topology, initialState) << std::endl;

    CpuSimulator simulator{std::move(initialState)};

    Options simOptions{
        .enableResourceTransfer = true
    };

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < simSteps; i++) {
        simulator.step(simOptions);
    }
    const State &finalState = simulator.getState();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << MapPrinter::printHexMapResources(topology, finalState) << std::endl;

    std::cout << "Simulation completed in " << duration.count() << " ms" << std::endl;
    std::cout << "Steps/second: " << (simSteps * 1000.0 / duration.count()) << std::endl;

    uint64_t checksum = computeChecksum(finalState.resources, topology.getStorageDimension());
    std::cout << "Map checksum: 0x" << std::hex << std::setw(16) << std::setfill('0') << checksum << std::dec << std::endl;

    return 0;
}
