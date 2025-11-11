//
// Template implementation for AxialRectangularMap
// This file is included by AxialRectangularMap.h
//

#include <stdexcept>
#include "../NotImplementedException.h"

template<typename CellData>
AxialRectangularMap<CellData>::AxialRectangularMap(int width, int height) :
        width(width), height(height) {
    assert(("Only square dimensions are supported.", width == height));
    maxCoordQ = width - 1;
    maxCoordR = maxCoordQ / 2 + maxCoordQ;
    storageDims = {maxCoordQ + 1, maxCoordR + 1};

    initializeStorageSize();
    initializeStoragePoints();
}

template<typename CellData>
void AxialRectangularMap<CellData>::initializeStorageSize() {
    int storageSizeWithPadding = (storageDims.first + 2) * (storageDims.second + 2);
    cells.resize(storageSizeWithPadding, CellData());
    validityMask.resize(storageSizeWithPadding, 0);
}

template<typename CellData>
void AxialRectangularMap<CellData>::initializeStoragePoints() {
    for (int r = 0; r < storageDims.second; r++) {
        for (int q = 0; q < storageDims.first; q++) {
            int storageIdx = (r + 1) * (storageDims.first + 2) + q + 1;

            if (!areCoordsOutOfBounds(q, r)) {
                // Set the cell as valid
                validityMask[storageIdx] = 1;
            }
        }
    }
}

template<typename CellData>
constexpr int AxialRectangularMap<CellData>::getWidth() const noexcept {
    return width;
}

template<typename CellData>
constexpr int AxialRectangularMap<CellData>::getHeight() const noexcept {
    return height;
}

template<typename CellData>
std::vector<Point *> AxialRectangularMap<CellData>::getNeighbors(const Point &point) {
    // The neighbor indices are different for odd and even columns
    const auto &axialDirectionVectors = (point.getY() % 2 == 0) ? axialDirectionVectorsEven : axialDirectionVectorsOdd;

    std::vector<Point *> neighbors;
    for (auto directionVector: axialDirectionVectors) {
        int newQ = directionVector[0] + point.getX();
        int newR = directionVector[1] + point.getY();
        if (!areCoordsOutOfBounds(newQ, newR)) {
            // Note: This returns nullptr as we no longer store Point objects
            neighbors.push_back(nullptr);
        }
    }

    return neighbors;
}

template<typename CellData>
bool AxialRectangularMap<CellData>::areCoordsOutOfBounds(int q, int r) const {
    int offset = maxCoordQ / 2;

    int offsetTop = q / 2;
    int offsetBottom = offset - offsetTop;

    if (q < 0 || q > maxCoordQ) {
        return true;
    }

    if (r < offsetTop || r > maxCoordR - offsetBottom) {
        return true;
    }
    return false;
}

template<typename CellData>
double AxialRectangularMap<CellData>::euclideanDistance(const Point &lhs, const Point &rhs) const {
    throw NotImplementedException();
}

template<typename CellData>
constexpr std::pair<int, int> AxialRectangularMap<CellData>::getMaxCoords() const noexcept {
    return {maxCoordQ, maxCoordR};
}

template<typename CellData>
constexpr std::pair<int, int> AxialRectangularMap<CellData>::getStorageDims() const noexcept {
    return storageDims;
}
