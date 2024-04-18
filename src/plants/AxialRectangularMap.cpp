//
// Created by lada on 1/31/22.
//

#include <memory>
#include <vector>
#include <stdexcept>
#include "AxialRectangularMap.h"
#include "../NotImplementedException.h"
#include "plants/iterators/AxialRectangularMapIterator.h"


AxialRectangularMap::AxialRectangularMap(std::size_t width, std::size_t height) :
        width(width), height(height) {
    widthStorageOffset = (height - 1) / 2;
    initializeStorageSize();
    initializeStoragePoints();
}

void AxialRectangularMap::initializeStorageSize() {
    storage.resize(height);
    for (int r = 0; r < height; r++) {
        storage[r].reserve(width);
    }
}

void AxialRectangularMap::initializeStoragePoints() {
    for (int r = 0; r < height; r++) {
//        auto widthWithOffset = widthStorageOffset + width;
        for (int q = 0; q < width; q++) {
            // For simplicity initialize even the empty space.
//            if (areCoordsOutOfBounds(q, r)) {
            storage[r][q] = Point{q, r};
            validPoints.push_back(storage[r][q]);
//            } else {
//                storage[r][q] = nullptr;
//            }
        }
    }
}

std::size_t AxialRectangularMap::getWidth() const {
    return width;
}

std::size_t AxialRectangularMap::getHeight() const {
    return height;
}

Point *AxialRectangularMap::getPoint(int x, int y) {
    if (areCoordsOutOfBounds(x, y)) {
        throw std::out_of_range("Indices q=" + std::to_string(x) +
                                " r=" + std::to_string(y) + " are out of range.");
    }
    return &storage[y][x];
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
    if (r < 0 || r >= height)
        return true;
    if (q < 0 || q >= width)
        return true;
    return false;
//    if (q < widthStorageOffset - r / 2)
//        return true;
//    if (q >= width + widthStorageOffset - r / 2)
//        return true;
//    return false;
}

double AxialRectangularMap::euclideanDistance(const Point &lhs, const Point &rhs) const {
    throw NotImplementedException();
}

std::vector<Point> &AxialRectangularMap::getPoints() {
    return validPoints;
}

