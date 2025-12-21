#include <vector>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <memory>

#include "simulation/ISimulator.h"
#include "simulation/Options.h"
#include "simulation/GridTopology.h"
#include "simulation/State.h"
#include "simulation/MapPrinter.h"
#include "simulation/SimulatorFactory.h"
#include "simulation/initializers/Initializers.h"
#include "simulation/initializers/PolicyApplication.h"

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
    using namespace initializers;

    OffsetCoord center{topology.width / 2, topology.height / 2};

    StateInitializer initializer{
        // Set all cells to Cell type
        //PolicyApplication{FullGrid{}, SetCellType{CellState::Cell}},
        PolicyApplication{CircleRegion{center, 1}, SetCellType{CellState::Cell}},     
        // Set resources at source cell
        PolicyApplication{
            SingleCell{center},
            SetResource{FixedAmount{1000.0f}}
        }
    };

    return initializer.initialize(topology);
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

    // Backend is set at compile time via -DTARGET_BACKEND=<backend>
    std::cout << "Using backend: " << SimulatorFactory::getBackendName() << std::endl;
    std::unique_ptr<ISimulator> simulatorPtr = SimulatorFactory::create(std::move(initialState));

    Options simOptions{
        .enableResourceTransfer = true,
        .enableCellMultiplication = true
    };

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < simSteps; i++) {
        simulatorPtr->step(simOptions);
    }
    const State &finalState = simulatorPtr->getState();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << MapPrinter::printHexMapResources(topology, finalState) << std::endl;
    std::cout << MapPrinter::printHexMapCellTypes(topology, finalState) << std::endl;

    std::cout << "Simulation completed in " << duration.count() << " ms" << std::endl;
    std::cout << "Steps/second: " << (simSteps * 1000.0 / duration.count()) << std::endl;

    uint64_t checksum = computeChecksum(finalState.plantSugar, topology.getStorageDimension());
    std::cout << "Map checksum: 0x" << std::hex << std::setw(16) << std::setfill('0') << checksum << std::dec << std::endl;

    return 0;
}
