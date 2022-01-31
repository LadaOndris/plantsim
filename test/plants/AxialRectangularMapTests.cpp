#include <gtest/gtest.h>
#include <memory>
#include "plants/AxialRectangularMap.h"

class AxialRectangularMapTest : public ::testing::Test {
protected:
public:
    AxialRectangularMapTest() : map(width, height) {};

protected:
    void SetUp() override {
        map = AxialRectangularMap(width, height);
    }

    std::size_t width = 7;
    std::size_t height = 7;
    AxialRectangularMap map;
};

TEST_F(AxialRectangularMapTest, CorrectSize) {
    auto givenWidth = map.getWidth();
    auto givenHeight = map.getHeight();

    ASSERT_EQ(width, givenWidth);
    ASSERT_EQ(height, givenHeight);
}

TEST_F(AxialRectangularMapTest, ElementAccessUpperLeft) {
    // Top left corner
    int q = 3;
    int r = 0;
    auto point = map.getPoint(q, r);

    ASSERT_EQ(q, point->getX());
    ASSERT_EQ(r, point->getY());
}


TEST_F(AxialRectangularMapTest, ElementAccessBottomLeft) {
    // Bottom left corner
    int q = 0;
    int r = 6;
    auto point = map.getPoint(q, r);

    ASSERT_EQ(q, point->getX());
    ASSERT_EQ(r, point->getY());
}

TEST_F(AxialRectangularMapTest, ElementAccessOutOfBounds) {
    int q = 9;
    int r = 2; // Out of bounds value

    EXPECT_THROW(map.getPoint(q, r), std::out_of_range);

}

TEST_F(AxialRectangularMapTest, NeighboursInside) {
    int q = 5;
    int r = 2;
    auto point = map.getPoint(q, r);
    auto neighbors = map.getNeighbors(point);

    // Each hexagon has 6 neighbors
    int neighborsCount = 6;
    ASSERT_EQ(neighborsCount, neighbors.size());

    int pointNeighbors[6][2] = {{6, 2},
                                {5, 3},
                                {4, 3},
                                {4, 2},
                                {5, 1},
                                {6, 1}};
    for (int i = 0; i < neighborsCount; i++) {
        auto pointQ = neighbors[i]->getX();
        auto pointR = neighbors[i]->getY();

        ASSERT_EQ(pointNeighbors[i][0], pointQ);
        ASSERT_EQ(pointNeighbors[i][1], pointR);
    }
}

TEST_F(AxialRectangularMapTest, NeighboursOnBorder) {
    int q = 5;
    int r = 0;
    auto point = map.getPoint(q, r);
    auto neighbors = map.getNeighbors(point);

    // Each hexagon on a border has 4 neighbors (not at the corners)
    int neighborsCount = 4;
    ASSERT_EQ(neighborsCount, neighbors.size());

    int pointNeighbors[4][2] = {{6, 0},
                                {5, 1},
                                {4, 1},
                                {4, 0}};
    for (int i = 0; i < neighborsCount; i++) {
        auto pointQ = neighbors[i]->getX();
        auto pointR = neighbors[i]->getY();

        ASSERT_EQ(pointNeighbors[i][0], pointQ);
        ASSERT_EQ(pointNeighbors[i][1], pointR);
    }
}

