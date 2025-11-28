#include "simulation/State.h"
#include "simulation/Options.h"
#include "simulation/ISimulator.h"
#include "simulation/cpu/CpuSimulator.h"
#include "simulation/sycl/SyclSimulator.h"

#include <gtest/gtest.h>
#include <memory>
#include <functional>

/**
 * @brief Factory function type for creating simulators.
 */
using SimulatorFactory = std::function<std::unique_ptr<ISimulator>(State)>;

/**
 * @brief Class that holds simulator factory for parametric tests.
 */
class ResourceTransferParams {
public:
    std::string name;
    SimulatorFactory factory;

    ResourceTransferParams(std::string name, SimulatorFactory factory)
        : name(std::move(name)), factory(std::move(factory)) {}

    std::unique_ptr<ISimulator> createSimulator(State initialState) const {
        return factory(std::move(initialState));
    }
};

// For test naming
std::ostream& operator<<(std::ostream& os, const ResourceTransferParams& param) {
    return os << param.name;
}

class ResourceTransferFixture : public ::testing::TestWithParam<ResourceTransferParams> {
protected:
    State createTestState() {
        const int width = 10;
        const int height = 10;
        const size_t totalCells = width * height;

        std::vector<int> resources(totalCells, 0);
        std::vector<int> cellTypes(totalCells, 0); // Air
        std::vector<std::pair<int, int>> neighborOffsets = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

        // Set up source cell with resources
        const int r = 1;
        const int q = 1;
        const int sourceIdx = r * width + q;
        resources[sourceIdx] = 1;
        cellTypes[sourceIdx] = 1; // Cell

        // Set up neighboring cell (right neighbor)
        const int neighborIdx = r * width + (q + 1);
        cellTypes[neighborIdx] = 1; // Cell

        return State(width, height, resources, cellTypes, neighborOffsets);
    }
};

TEST_P(ResourceTransferFixture, SingleStep) {
    Options options {
        .enableResourceTransfer = true 
    };

    ResourceTransferParams param = GetParam();
    State initialState = createTestState();

    auto simulator = param.createSimulator(initialState);

    simulator->step(options);
    const State &finalState = simulator->getState();

    // Verify resource conservation
    int initialTotal = 0;
    int finalTotal = 0;
    for (size_t i = 0; i < initialState.totalCells(); ++i) {
        initialTotal += initialState.resources[i];
        finalTotal += finalState.resources[i];
    }
    ASSERT_EQ(initialTotal, finalTotal) << "Resources should be conserved";

    // Check that the source cell has transferred its resource to the neighbor
    const int sourceIdx = 1 * initialState.width + 1;
    const int neighborIdx = 1 * initialState.width + 2;
    ASSERT_EQ(finalState.resources[neighborIdx], 1);
    ASSERT_EQ(finalState.resources[sourceIdx], 0);
}

INSTANTIATE_TEST_SUITE_P(
    ResourceTransferTests, 
    ResourceTransferFixture,
    ::testing::Values(
        ResourceTransferParams{"CpuSimulator", [](State s) { 
            return std::make_unique<CpuSimulator>(std::move(s)); 
        }},
        ResourceTransferParams{"SyclSimulator", [](State s) { 
            return std::make_unique<SyclSimulator>(std::move(s)); 
        }}
    ),
    [](const ::testing::TestParamInfo<ResourceTransferParams>& info) {
        return info.param.name;
    }
);