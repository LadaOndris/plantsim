
#include "simulation/GridTopology.h"

#include <gtest/gtest.h>

TEST(GridTopologyTest, TotalCellsCalculation) {
    GridTopology topology(7, 7);
    EXPECT_EQ(topology.totalCells(), 49);
}
struct StorageTestParam {
    int cols;
    int rows;
    int expStorageCols;
    int expStorageRows;
};

class GridTopologyStorageTest : public ::testing::TestWithParam<StorageTestParam> {};

TEST_P(GridTopologyStorageTest, StorageDimension) {
    const auto p = GetParam();
    GridTopology topology(p.cols, p.rows);
    EXPECT_EQ(topology.getStorageDimension(), std::make_pair(p.expStorageCols, p.expStorageRows));
}

INSTANTIATE_TEST_SUITE_P(
    StorageDimensionCases,
    GridTopologyStorageTest,
    ::testing::Values(
        StorageTestParam{7, 7, 10, 7},
        StorageTestParam{1, 3, 2, 3},
        StorageTestParam{7, 3, 8, 3},
        StorageTestParam{1, 1, 1, 1},
        StorageTestParam{4, 4, 5, 4}
    )
);

struct AxialToStorageCoordTestParam {
    std::pair<int, int> axial;
    std::pair<int, int> storage;
    int width;
    int height;
};

class GridTopologyAxialToStorageCoordTest : public ::testing::TestWithParam<AxialToStorageCoordTestParam> {};

TEST_P(GridTopologyAxialToStorageCoordTest, AxialToStorageCoord) {
    const auto p = GetParam();
    GridTopology topology(p.width, p.height);
    EXPECT_EQ(topology.axialToStorageCoord(p.axial), p.storage);
}

INSTANTIATE_TEST_SUITE_P(
    AxialToStorageCoordCases,
    GridTopologyAxialToStorageCoordTest,
    ::testing::Values(
        AxialToStorageCoordTestParam{{0, 0}, {3, 0}, 7, 7},
        AxialToStorageCoordTestParam{{-3, 6}, {0, 6}, 7, 7},
        AxialToStorageCoordTestParam{{6, 1}, {9, 1}, 7, 7},
        AxialToStorageCoordTestParam{{1, 6}, {4, 6}, 7, 7},
        AxialToStorageCoordTestParam{{0, 0}, {0, 0}, 1, 1},
        AxialToStorageCoordTestParam{{-1, 2}, {0, 2}, 1, 3},
        AxialToStorageCoordTestParam{{0, 1}, {1, 1}, 1, 3}
    )
);

class GridTopologyStorageToAxialCoordTest : public ::testing::TestWithParam<AxialToStorageCoordTestParam> {};

TEST_P(GridTopologyStorageToAxialCoordTest, StorageToAxialCoord) {
    const auto p = GetParam();
    GridTopology topology(p.width, p.height);
    EXPECT_EQ(topology.storageToAxialCoord(p.storage), p.axial);
}

INSTANTIATE_TEST_SUITE_P(
    StorageToAxialCoordCases,
    GridTopologyStorageToAxialCoordTest,
    ::testing::Values(
        AxialToStorageCoordTestParam{{0, 0}, {3, 0}, 7, 7},
        AxialToStorageCoordTestParam{{-3, 6}, {0, 6}, 7, 7},
        AxialToStorageCoordTestParam{{6, 1}, {9, 1}, 7, 7},
        AxialToStorageCoordTestParam{{1, 6}, {4, 6}, 7, 7},
        AxialToStorageCoordTestParam{{0, 0}, {0, 0}, 1, 1},
        AxialToStorageCoordTestParam{{-1, 2}, {0, 2}, 1, 3},
        AxialToStorageCoordTestParam{{0, 1}, {1, 1}, 1, 3}
    )
);

TEST_P(GridTopologyAxialToStorageCoordTest, ToIndex) {
    const auto p = GetParam();
    GridTopology topology(p.width, p.height);
    auto [q, r] = p.axial;
    auto [x, y] = p.storage;
    int expectedIndex = y * topology.getStorageDimension().first + x;
    EXPECT_EQ(topology.toIndex(r, q), expectedIndex);
}

INSTANTIATE_TEST_SUITE_P(
    ToIndexCases,
    GridTopologyAxialToStorageCoordTest,
    ::testing::Values(
        AxialToStorageCoordTestParam{{0, 0}, {3, 0}, 7, 7},
        AxialToStorageCoordTestParam{{-3, 6}, {0, 6}, 7, 7},
        AxialToStorageCoordTestParam{{6, 1}, {9, 1}, 7, 7},
        AxialToStorageCoordTestParam{{1, 6}, {4, 6}, 7, 7},
        AxialToStorageCoordTestParam{{0, 0}, {0, 0}, 1, 1},
        AxialToStorageCoordTestParam{{-1, 2}, {0, 2}, 1, 3},
        AxialToStorageCoordTestParam{{0, 1}, {1, 1}, 1, 3}
    )
);
