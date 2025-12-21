#include "simulation/GridTopology.h"
#include "simulation/State.h"
#include "simulation/Options.h"
#include "simulation/ISimulator.h"
#include "simulation/cpu/CpuSimulator.h"
// #include "simulation/sycl/SyclSimulator.h"  // Commented out - SYCL backend not enabled
#include "simulation/MapPrinter.h"

#include <gtest/gtest.h>
#include <memory>
#include <functional>

/**
 * @brief Factory function type for creating simulators.
 */
using SimulatorFactory = std::function<std::unique_ptr<ISimulator>(State, const Options&)>;

/**
 * @brief Class that holds simulator factory for parametric tests.
 */
class ResourceTransferParams {
public:
    std::string name;
    SimulatorFactory factory;

    ResourceTransferParams(std::string name, SimulatorFactory factory)
        : name(std::move(name)), factory(std::move(factory)) {}

    std::unique_ptr<ISimulator> createSimulator(State initialState, const Options& options) const {
        return factory(std::move(initialState), options);
    }
};

// For test naming
std::ostream& operator<<(std::ostream& os, const ResourceTransferParams& param) {
    return os << param.name;
}

class ResourceTransferFixture : public ::testing::TestWithParam<ResourceTransferParams> {
protected:
    State createTestState(const GridTopology &topology) {
        const int width = topology.width;
        const int height = topology.height;
        const size_t totalCells = width * height;

        std::vector<float> plantSugar(totalCells, 0.0f);
        std::vector<float> plantWater(totalCells, 0.0f);
        std::vector<float> plantMineral(totalCells, 0.0f);
        std::vector<float> soilWater(totalCells, 0.0f);
        std::vector<float> soilMineral(totalCells, 0.0f);
        std::vector<int> cellTypes(totalCells, 0); // Air

        // Set up source cell with resources
        const AxialCoord cell{.q=1, .r=1};
        const int sourceIdx = cell.asFlat(topology.getDimension());
        plantSugar[sourceIdx] = 1.0f;
        cellTypes[sourceIdx] = 1; // Cell type

        // Set up neighboring cell (right neighbor)
        const AxialCoord neighbor{.q=2, .r=1};
        const int neighborIdx = neighbor.asFlat(topology.getDimension());
        cellTypes[neighborIdx] = 1; // Cell type

        // State should contain the storage data.
        auto storedPlantSugar = store(plantSugar, width, height, 0.0f);
        auto storedPlantWater = store(plantWater, width, height, 0.0f);
        auto storedPlantMineral = store(plantMineral, width, height, 0.0f);
        auto storedSoilWater = store(soilWater, width, height, 0.0f);
        auto storedSoilMineral = store(soilMineral, width, height, 0.0f);
        auto storedCellTypes = store(cellTypes, width, height, -1);

        return State(width, height, 
                     std::move(storedCellTypes),
                     std::move(storedSoilWater),
                     std::move(storedSoilMineral),
                     std::move(storedPlantSugar),
                     std::move(storedPlantWater),
                     std::move(storedPlantMineral));
    }
};

TEST_P(ResourceTransferFixture, SingleStep) {
    Options options {
        .enableResourceTransfer = true 
    };

    ResourceTransferParams param = GetParam();
    GridTopology topology(5, 5);
    State initialState = createTestState(topology);

    std::cout << MapPrinter::printHexMapCellTypes(topology, initialState) << std::endl;
    std::cout << MapPrinter::printHexMapResources(topology, initialState) << std::endl;

    auto simulator = param.createSimulator(std::move(initialState), options);

    simulator->step(options);
    const State &finalState = simulator->getState();

    std::cout << MapPrinter::printHexMapCellTypes(topology, finalState) << std::endl;
    std::cout << MapPrinter::printHexMapResources(topology, finalState) << std::endl;

    // Verify resource conservation
    float initialTotal = 0.0f;
    float finalTotal = 0.0f;
    for (size_t i = 0; i < initialState.totalCells(); ++i) {
        initialTotal += initialState.plantSugar[i];
        finalTotal += finalState.plantSugar[i];
    }
    ASSERT_FLOAT_EQ(initialTotal, finalTotal) << "Resources should be conserved";

    // Check that diffusion has occurred between the two plant cells
    const AxialCoord cell {.q=1, .r=1};
    const AxialCoord neighbor{.q=2, .r=1};

    const int sourceIdx = topology.axialToStorageCoord(cell).asFlat(topology.getStorageDimension());
    const int neighborIdx = topology.axialToStorageCoord(neighbor).asFlat(topology.getStorageDimension());

    // With diffusion, sugar should have spread from source to neighbor
    // The exact values depend on the transport rate, but neighbor should now have some sugar
    ASSERT_GT(finalState.plantSugar[neighborIdx], 0.0f) << "Neighbor should receive some sugar";
    ASSERT_LT(finalState.plantSugar[sourceIdx], 1.0f) << "Source should have less sugar after transfer";
}

INSTANTIATE_TEST_SUITE_P(
    ResourceTransferTests, 
    ResourceTransferFixture,
    ::testing::Values(
        ResourceTransferParams{"CpuSimulator", [](State s, const Options& opts) { 
            return std::make_unique<CpuSimulator>(std::move(s), opts); 
        }}
        // SyclSimulator test commented out - SYCL backend not enabled
        // ,ResourceTransferParams{"SyclSimulator", [](State s, const Options& opts) { 
        //     return std::make_unique<SyclSimulator>(std::move(s), opts); 
        // }}
    ),
    [](const ::testing::TestParamInfo<ResourceTransferParams>& info) {
        return info.param.name;
    }
);