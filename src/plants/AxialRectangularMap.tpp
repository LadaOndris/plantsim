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
}

template<typename CellData>
void AxialRectangularMap<CellData>::initializeStoragePoints() {
    for (int r = 0; r < storageDims.second; r++) {
        for (int q = 0; q < storageDims.first; q++) {
            int storageIdx = (r + 1) * (storageDims.first + 2) + q + 1;

            cells[storageIdx].valid = static_cast<uint8_t>(!areCoordsOutOfBounds(q, r));

        }
    }
}

template<typename CellData>
int AxialRectangularMap<CellData>::getWidth() const {
    return width;
}

template<typename CellData>
int AxialRectangularMap<CellData>::getHeight() const {
    return height;
}

template<typename CellData>
Point *AxialRectangularMap<CellData>::getPoint(int x, int y) {
    // Note: This method is deprecated as we no longer store Point objects directly
    // Keeping for compatibility
    if (areCoordsOutOfBounds(x, y)) {
        throw std::out_of_range("Indices q=" + std::to_string(x) +
                                " r=" + std::to_string(y) + " are out of range.");
    }
    throw std::logic_error("getPoint() is deprecated. Use getCellAt() instead.");
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
std::vector<Point *> &AxialRectangularMap<CellData>::getPoints() {
    return validPoints;
}

template<typename CellData>
std::pair<int, int> AxialRectangularMap<CellData>::getMaxCoords() const {
    return {maxCoordQ, maxCoordR};
}

template<typename CellData>
const std::vector<std::pair<int, int>> &AxialRectangularMap<CellData>::getNeighborOffsets() const {
    return neighborOffsets;
}

template<typename CellData>
std::pair<int, int> AxialRectangularMap<CellData>::getStorageDims() const {
    return storageDims;
}
