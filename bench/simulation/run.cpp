#include <memory>
#include <vector>
#include <iostream>
#include <chrono>
#include <iomanip>

#include "simulation/Simulator.h"
#include "simulation/SimulatorOptions.h"
#include "simulation/CellState.h"
#include "plants/WorldState.h"
#include "plants/AxialRectangularMap.h"

std::unique_ptr<WorldState> initializeWorld() {
    std::vector<std::shared_ptr<Process>> processes{};
    auto map = std::make_shared<AxialRectangularMap<CellState>>(200, 200);

    auto worldState{std::make_unique<WorldState>(map, processes)};
    return worldState;
}

uint64_t computeChecksum(const std::vector<CellState> &cells, int storageDimsFirst, int storageDimsSecond) {
    uint64_t checksum = 0;
    for (int r = 0; r < storageDimsSecond; r++) {
        for (int q = 0; q < storageDimsFirst; q++) {
            int idx = (r + 1) * (storageDimsFirst + 2) + q + 1;
            checksum = checksum * 31 + static_cast<int>(cells[idx].type);
        }
    }
    return checksum;
}

void printMapCorner(const AxialRectangularMap<CellState> &map, int cornerSize) {
    const auto &cells = map.getCells();
    auto storageDims = map.getStorageDims();

    std::cout << "Map corner (" << cornerSize << "x" << cornerSize << "):\n";
    for (int r = 0; r < cornerSize && r < storageDims.second; r++) {
        for (int q = 0; q < cornerSize && q < storageDims.first; q++) {
            int idx = (r + 1) * (storageDims.first + 2) + q + 1;
            char symbol = (cells[idx].type == CellState::Type::Cell) ? 'C' : '.';
            std::cout << symbol;
        }
        std::cout << '\n';
    }
    std::cout << '\n';
}

int main() {
    auto worldState = initializeWorld();
    Simulator simulator(*worldState);
    auto &map = worldState->getMap();
    auto storageDims = map.getStorageDims();

    SimulatorOptions simOptions{};
    const int simSteps = 2000;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < simSteps; i++) {
        simulator.step(simOptions);
    }
    simulator.updateCurrentState();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Simulation completed in " << duration.count() << " ms" << std::endl;

    std::cout << "Steps/second: " << (simSteps * 1000.0 / duration.count()) << std::endl;

    printMapCorner(map, 20);

    uint64_t checksum = computeChecksum(map.getCells(), storageDims.first, storageDims.second);
    std::cout << "Map checksum: 0x" << std::hex << std::setw(16) << std::setfill('0') << checksum << std::dec << std::endl;

    return 0;
}
