//
// Created by lada on 1/31/22.
//

#include <memory>
#include <vector>
#include <stdexcept>
#include <cassert>
#include "AxialRectangularMap.h"
#include "../NotImplementedException.h"
#include "plants/iterators/AxialRectangularMapIterator.h"


AxialRectangularMap::AxialRectangularMap(int width, int height) :
        width(width), height(height) {
    assert(("Only square dimensions are supported.", width == height));
    maxCoordQ = width - 1;
    maxCoordR = maxCoordQ / 2 + maxCoordQ;
    storageDims = {maxCoordQ + 1, maxCoordR + 1};

    initializeStorageSize();
    initializeStoragePoints();
}

void AxialRectangularMap::initializeStorageSize() {
    storage.reserve(storageDims.first * storageDims.second);
}

void AxialRectangularMap::initializeStoragePoints() {
    for (int r = 0; r < storageDims.second; r++) {
        for (int q = 0; q < storageDims.first; q++) {
            storage[r * storageDims.first + q] = Point{q, r};

            if (!areCoordsOutOfBounds(q, r)) {
                validPoints.push_back(&storage[r * storageDims.first + q]);
            }
        }
    }
}

int AxialRectangularMap::getWidth() const {
    return width;
}

int AxialRectangularMap::getHeight() const {
    return height;
}

Point *AxialRectangularMap::getPoint(int x, int y) {
    // q = x;
    // r = y;
    if (areCoordsOutOfBounds(x, y)) {
        throw std::out_of_range("Indices q=" + std::to_string(x) +
                                " r=" + std::to_string(y) + " are out of range.");
    }
    return &storage[y * storageDims.first + x];
}

std::vector<Point *> AxialRectangularMap::getNeighbors(const Point &point) {
    // The neighbor indices are different odd and even columns
    const auto &axialDirectionVectors = (point.getY() % 2 == 0) ? axialDirectionVectorsEven : axialDirectionVectorsOdd;

    std::vector<Point *> neighbors;
    for (auto directionVector: axialDirectionVectors) {
        int newQ = directionVector[0] + point.getX();
        int newR = directionVector[1] + point.getY();
        if (!areCoordsOutOfBounds(newQ, newR)) {
            auto neighbor = getPoint(newQ, newR);
            neighbors.push_back(neighbor);
        }
    }

    return neighbors;
}

bool AxialRectangularMap::areCoordsOutOfBounds(int q, int r) const {
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

double AxialRectangularMap::euclideanDistance(const Point &lhs, const Point &rhs) const {
    throw NotImplementedException();
}

std::vector<Point *> &AxialRectangularMap::getPoints() {
    return validPoints;
}

std::pair<int, int> AxialRectangularMap::getMaxCoords() const {
    return {maxCoordQ, maxCoordR};
}

