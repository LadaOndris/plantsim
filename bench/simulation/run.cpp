#include <memory>
#include <vector>
#include <iostream>
#include <chrono>

#include "simulation/Simulator.h"
#include "simulation/SimulatorOptions.h"
#include "plants/WorldState.h"
#include "plants/AxialRectangularMap.h"

std::unique_ptr<WorldState> initializeWorld() {
    std::vector<std::shared_ptr<Process>> processes{};
    auto map = std::make_shared<AxialRectangularMap>(200, 200);

    auto worldState{std::make_unique<WorldState>(map, processes)};
    return worldState;
}

int main() {
    auto worldState = initializeWorld();
    Simulator simulator(*worldState);

    SimulatorOptions simOptions{};

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 100; i++) {
        simulator.step(simOptions);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Simulation completed in " << duration.count() << " ms" << std::endl;

    return 0;
}
