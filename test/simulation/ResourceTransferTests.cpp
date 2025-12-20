#include "simulation/GridTopology.h"
#include "simulation/State.h"
#include "simulation/Options.h"
#include "simulation/ISimulator.h"
#include "simulation/cpu/CpuSimulator.h"
#include "simulation/sycl/SyclSimulator.h"
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
        auto storedResources = store(resources, width, height, 0);
        auto storedCellTypes = store(cellTypes, width, height, -1);

        return State(width, height, 
                     std::vector<float>(storedResources.begin(), storedResources.end()), 
                     storedCellTypes);
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
    int initialTotal = 0;
    int finalTotal = 0;
    for (size_t i = 0; i < initialState.totalCells(); ++i) {
        initialTotal += initialState.resources[i];
        finalTotal += finalState.resources[i];
    }
    ASSERT_EQ(initialTotal, finalTotal) << "Resources should be conserved";

    // Check that the source cell has transferred its resource to the neighbor
    const AxialCoord cell {.q=1, .r=1};
    const AxialCoord neighbor{.q=2, .r=1};

    const int sourceIdx = topology.axialToStorageCoord(cell).asFlat(topology.getStorageDimension());
    const int neighborIdx = topology.axialToStorageCoord(neighbor).asFlat(topology.getStorageDimension());

    ASSERT_EQ(finalState.resources[neighborIdx], 1);
    ASSERT_EQ(finalState.resources[sourceIdx], 0);
}

INSTANTIATE_TEST_SUITE_P(
    ResourceTransferTests, 
    ResourceTransferFixture,
    ::testing::Values(
        ResourceTransferParams{"CpuSimulator", [](State s, const Options& opts) { 
            return std::make_unique<CpuSimulator>(std::move(s), opts); 
        }},
        ResourceTransferParams{"SyclSimulator", [](State s, const Options& opts) { 
            return std::make_unique<SyclSimulator>(std::move(s), opts); 
        }}
    ),
    [](const ::testing::TestParamInfo<ResourceTransferParams>& info) {
        return info.param.name;
    }
);