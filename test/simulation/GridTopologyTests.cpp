#include "simulation/GridTopology.h"

#include <gtest/gtest.h>

TEST(GridTopologyTest, TotalCellsCalculation) {
    GridTopology topology(7, 7);
    EXPECT_EQ(topology.totalCells(), 49);
}

struct StorageTestParam {
    int cols;
    int rows;
    StorageCoord expected;
};

class GridTopologyStorageTest : public ::testing::TestWithParam<StorageTestParam> {};

TEST_P(GridTopologyStorageTest, StorageDimension) {
    const auto p = GetParam();
    GridTopology topology(p.cols, p.rows);
    EXPECT_EQ(topology.storageDim, p.expected);
}

INSTANTIATE_TEST_SUITE_P(
    StorageDimensionCases,
    GridTopologyStorageTest,
    ::testing::Values(
        StorageTestParam{7, 7, {10, 7}},
        StorageTestParam{1, 3, {2, 3}},
        StorageTestParam{7, 3, {8, 3}},
        StorageTestParam{1, 1, {1, 1}},
        StorageTestParam{4, 4, {5, 4}}
    )
);

struct AxialToStorageCoordTestParam {
    AxialCoord axial;
    StorageCoord storage;
    int width;
    int height;
};

class GridTopologyAxialToStorageCoordTest : public ::testing::TestWithParam<AxialToStorageCoordTestParam> {};

TEST_P(GridTopologyAxialToStorageCoordTest, AxialToStorageCoord) {
    const auto p = GetParam();
    GridTopology topology(p.width, p.height);
    EXPECT_EQ(topology.toStorageCoord(p.axial), p.storage);
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
    EXPECT_EQ(topology.toAxialCoord(p.storage), p.axial);
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
    StorageCoord dim = topology.storageDim;
    int expectedIndex = p.storage.y * dim.x + p.storage.x;
    EXPECT_EQ(topology.toStorageIndex(p.axial), expectedIndex);
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

struct IsValidTestParam {
    AxialCoord axial;
    int width;
    int height;
    bool expected;
};

class GridTopologyIsValidTest : public ::testing::TestWithParam<IsValidTestParam> {};

TEST_P(GridTopologyIsValidTest, IsValid) {
    const auto p = GetParam();
    GridTopology topology(p.width, p.height);
    EXPECT_EQ(topology.isValid(p.axial), p.expected);
}

INSTANTIATE_TEST_SUITE_P(
    IsValidCases,
    GridTopologyIsValidTest,
    ::testing::Values(
        // 3x3 grid - valid cells
        IsValidTestParam{{0, 0}, 3, 3, true},   // row 0: q in [0, 2]
        IsValidTestParam{{1, 0}, 3, 3, true},
        IsValidTestParam{{2, 0}, 3, 3, true},
        IsValidTestParam{{0, 1}, 3, 3, true},   // row 1 (odd): q in [0, 2]
        IsValidTestParam{{1, 1}, 3, 3, true},
        IsValidTestParam{{2, 1}, 3, 3, true},
        IsValidTestParam{{-1, 2}, 3, 3, true},  // row 2: q in [-1, 1]
        IsValidTestParam{{0, 2}, 3, 3, true},
        IsValidTestParam{{1, 2}, 3, 3, true},
        
        // 3x3 grid - invalid cells (storage padding)
        IsValidTestParam{{-1, 0}, 3, 3, false}, // left of row 0
        IsValidTestParam{{-1, 1}, 3, 3, false}, // left of row 1
        IsValidTestParam{{2, 2}, 3, 3, false},  // right of row 2

        // 3x3 grid - invalid cells (out of storage col bounds)
        IsValidTestParam{{3, 0}, 3, 3, false},  // right of row 0
        IsValidTestParam{{3, 1}, 3, 3, false},  // right of row 1
        IsValidTestParam{{-2, 2}, 3, 3, false}, // left of row 2
        
        // 3x3 grid - invalid cells (out of storage row bounds)
        IsValidTestParam{{0, -1}, 3, 3, false},
        IsValidTestParam{{0, 3}, 3, 3, false},
        
        // 1x1 grid
        IsValidTestParam{{0, 0}, 1, 1, true},
        IsValidTestParam{{1, 0}, 1, 1, false},
        IsValidTestParam{{-1, 0}, 1, 1, false}
    )
);
