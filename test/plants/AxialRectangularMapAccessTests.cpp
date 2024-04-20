#include <gtest/gtest.h>
#include <memory>
#include "plants/AxialRectangularMap.h"

class AxialRectangularMapAccessTest : public ::testing::Test {
protected:
public:
    AxialRectangularMapAccessTest() : map(width, height) {};

protected:
    void SetUp() override {
        map = AxialRectangularMap(width, height);
    }

    std::size_t width = 4;
    std::size_t height = 4;
    AxialRectangularMap map;
};

TEST_F(AxialRectangularMapAccessTest, CorrectSizeInRowsAndColumns) {
    auto givenWidth = map.getWidth();
    auto givenHeight = map.getHeight();

    ASSERT_EQ(width, givenWidth);
    ASSERT_EQ(height, givenHeight);
}


TEST_F(AxialRectangularMapAccessTest, CorrectSizeInAxialCoords) {
    struct SizeToCoordCorrespondence {
        int mapSize;
        int maxQ;
        int maxR;
    };
    std::vector<SizeToCoordCorrespondence> sizeToMaxAxialCoordCorrespondences{
            {1, 0, 0},
            {2, 1, 1},
            {3, 2, 3},
            {4, 3, 4},
            {5, 4, 6},
            {6, 5, 7},
            {7, 6, 9},
    };
    for (const auto &correspondence: sizeToMaxAxialCoordCorrespondences) {
        AxialRectangularMap map{correspondence.mapSize, correspondence.mapSize};
        auto givenMaxCoords = map.getMaxCoords();

        ASSERT_EQ(correspondence.maxQ, givenMaxCoords.first);
        ASSERT_EQ(correspondence.maxR, givenMaxCoords.second);
    }
}

TEST_F(AxialRectangularMapAccessTest, CornerElementAccess) {
    std::vector<std::pair<int, int>> cornersCoords{
            {0, 0},
            {0, 3},
            {3, 1},
            {3, 4}
    };
    for (const auto &cornerCoords: cornersCoords) {
        int q = cornerCoords.first;
        int r = cornerCoords.second;

        auto point = map.getPoint(q, r);

        ASSERT_EQ(q, point->getX());
        ASSERT_EQ(r, point->getY());
    }
}


TEST_F(AxialRectangularMapAccessTest, ElementAccessOutOfBounds) {
    // Three out of bounds coords for each corner
    std::vector<std::pair<int, int>> outOfBoundsCoords{
            // Top left
            {0,  -1},
            {-1, -1},
            {-1, 0},
            // Top right
            {0,  4},
            {-1, 4},
            {-1, 3},
            // Bottom left
            {3,  0},
            {4,  0},
            {4,  1},
            // Bottom right
            {4,  4},
            {4,  5},
            {3,  5}
    };


    for (const auto &coords: outOfBoundsCoords) {
        int q = coords.first;
        int r = coords.second;

        EXPECT_THROW(map.getPoint(q, r), std::out_of_range);
    }
}

//TEST_F(AxialRectangularMapTest, NeighboursInside) {
//    int q = 5;
//    int r = 2;
//    auto point = map.getPoint(q, r);
//    auto neighbors = map.getNeighbors(point);
//
//    // Each hexagon has 6 neighbors
//    int neighborsCount = 6;
//    ASSERT_EQ(neighborsCount, neighbors.size());
//
//    int pointNeighbors[6][2] = {{6, 2},
//                                {5, 3},
//                                {4, 3},
//                                {4, 2},
//                                {5, 1},
//                                {6, 1}};
//    for (int i = 0; i < neighborsCount; i++) {
//        auto pointQ = neighbors[i]->getX();
//        auto pointR = neighbors[i]->getY();
//
//        ASSERT_EQ(pointNeighbors[i][0], pointQ);
//        ASSERT_EQ(pointNeighbors[i][1], pointR);
//    }
//}
//
//TEST_F(AxialRectangularMapTest, NeighboursOnBorder) {
//    int q = 5;
//    int r = 0;
//    auto point = map.getPoint(q, r);
//    auto neighbors = map.getNeighbors(point);
//
//    // Each hexagon on a border has 4 neighbors (not at the corners)
//    int neighborsCount = 4;
//    ASSERT_EQ(neighborsCount, neighbors.size());
//
//    int pointNeighbors[4][2] = {{6, 0},
//                                {5, 1},
//                                {4, 1},
//                                {4, 0}};
//    for (int i = 0; i < neighborsCount; i++) {
//        auto pointQ = neighbors[i]->getX();
//        auto pointR = neighbors[i]->getY();
//
//        ASSERT_EQ(pointNeighbors[i][0], pointQ);
//        ASSERT_EQ(pointNeighbors[i][1], pointR);
//    }
//}

