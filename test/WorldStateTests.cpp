
#include <gtest/gtest.h>
#include <memory>
#include "plants/WorldState.h"
#include "plants/Point.h"
#include "dummies/EmptyProcess.h"

TEST(WorldState, Indexer_PointsHaveCorrectCoords) {
    int width = 10;
    int height = 20;

    std::shared_ptr<Process> emptyProcess = std::make_shared<EmptyProcess>();
    std::vector<std::shared_ptr<Process>> processes;
    processes.push_back(std::move(emptyProcess));

    WorldState state(width, height, processes);

    int x = 2;
    int y = 5;
    auto pPoint = state[x * width + y];
    Point &point = *pPoint;

    ASSERT_EQ(x, point.getX());
    ASSERT_EQ(y, point.getY());
}


TEST(WorldState, GetPoint_PointsHaveCorrectCoords) {
    int width = 10;
    int height = 20;

    std::shared_ptr<Process> emptyProcess = std::make_shared<EmptyProcess>();
    std::vector<std::shared_ptr<Process>> processes;
    processes.push_back(std::move(emptyProcess));

    WorldState state(width, height, processes);

    int x = 2;
    int y = 5;
    auto pPoint = state.getPoint(x, y);
    Point &point = *pPoint;

    ASSERT_EQ(x, point.getX());
    ASSERT_EQ(y, point.getY());
}