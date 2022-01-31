
#include <gtest/gtest.h>
#include <memory>
#include <plants/AxialRectangularMap.h>
#include "plants/WorldState.h"
#include "plants/Point.h"
#include "../dummies/EmptyProcess.h"

TEST(WorldState, getTotalGenesCount) {
    int width = 10;
    int height = 20;

    std::shared_ptr<Process> emptyProcess = std::make_shared<EmptyProcess>();
    std::vector<std::shared_ptr<Process>> processes;
    processes.push_back(emptyProcess);
    processes.push_back(emptyProcess);

    auto map = std::make_shared<AxialRectangularMap>(width, height);
    WorldState state(map, processes);

    int genesCount = state.getTotalGenesCount();

    ASSERT_EQ(2 * emptyProcess->getGenesCount(), genesCount);
}
