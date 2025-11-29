
#include "simulation/MapPrinter.h"
#include "simulation/GridTopology.h"
#include <gtest/gtest.h>

OffsetCoord axialToOddr(const AxialCoord& hex) {
    int parity = hex.r & 1;
    int col = hex.q + (hex.r - parity) / 2;
    int row = hex.r;
    return OffsetCoord{col, row};
}

AxialCoord oddrToAxial(const OffsetCoord& hex) {
    int parity = hex.row & 1;
    int q = hex.col - (hex.row - parity) / 2;
    int r = hex.row;
    return AxialCoord{q, r};
}

std::vector<int> store(std::vector<int> data, int width, int height, int defaultFillValue = -1) {
    std::vector<int> storage;
    GridTopology topology(width, height);
    StorageCoord dim = topology.getStorageDimension();
    storage.resize(dim.x * dim.y, defaultFillValue);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int value = data[y * width + x];
            AxialCoord axial = oddrToAxial({x, y});
            StorageCoord storageCoord = topology.axialToStorageCoord(axial);
            storage[storageCoord.y * dim.x + storageCoord.x] = value;
        }
    }

    return storage;
}

TEST(MapPrinterTest, PrintHexMapResources) {
    const int width = 5;
    const int height = 4;
    GridTopology topology(width, height);

    std::vector<int> resources = {
        0, 1, 2, 10, 5,
        3, 0, 0, 7, 9,
        0, 0, 0, 0, 0,
        8, 6, 4, 2, 1
    };
    std::vector<int> storedResources = store(resources, width, height, 0);

    std::string expectedOutput =
        ". 1 2 + 5 \n"
        " 3 . . 7 9 \n"
        ". . . . . \n"
        " 8 6 4 2 1 \n";

    std::string output = MapPrinter::printHexMap<int>(
        topology, storedResources,
        [](int v) -> char {
            if (v == 0)
                return '.';
            if (v <= 9)
                return static_cast<char>('0' + v);
            return '+';
        });

    EXPECT_EQ(output, expectedOutput);
}

TEST(MapPrinterTest, PrintHexMapCellTypes) {
    const int width = 3;
    const int height = 3;
    GridTopology topology(width, height);

    std::vector<int> cellTypes = {
        0, 1, 2,
        1, 0, 2,
        2, 1, 0
    };
    std::vector<int> storedCellTypes = store(cellTypes, width, height, -1);

    std::string expectedOutput =
        "A B C \n"
        " B A C \n"
        "C B A \n";

    std::string output = MapPrinter::printHexMap<int>(
        topology, storedCellTypes,
        [](int v) -> char {
            switch (v) {
                case -1: return '-';
                case 0: return 'A';
                case 1: return 'B';
                case 2: return 'C';
                default: return '?';
            }
        });

    EXPECT_EQ(output, expectedOutput);
}


TEST(MapPrinterTest, PrintStorageMap) {
    const int width = 4;
    const int height = 4;
    GridTopology topology(width, height);

    std::vector<int> data = {
        1, 0, 0, 1,
        0, 1, 1, 0,
        0, 1,0,1,
        1,0,1,0
    };
    std::vector<int> storedData = store(data, width, height, -1);

    std::string expectedOutput =
        "- + 0 0 + \n"
        "- 0 + + 0 \n"
        "0 + 0 + - \n"
        "+ 0 + 0 - \n";

    std::string output = MapPrinter::printStorage<int>(
        topology, storedData,
        [](int v) -> char {
            if (v == -1)
                return '-';
            if (v == 0)
                return '0';
            if (v > 0)
                return '+';
            return '?';
        });

    EXPECT_EQ(output, expectedOutput);
}