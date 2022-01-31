//
// Created by lada on 1/31/22.
//

#include <memory>
#include <vector>
#include <stdexcept>
#include "AxialRectangularMap.h"
#include "../NotImplementedException.h"


AxialRectangularMap::AxialRectangularMap(std::size_t width, std::size_t height) :
        width(width), height(height) {
    widthStorageOffset = (height - 1) / 2;
    initializeStorageSize();
    initializeStoragePoints();
}

void AxialRectangularMap::initializeStorageSize() {
    storage.resize(height);
    for (int r = 0; r < height; r++) {
        storage[r].resize(widthStorageOffset + width);
    }
}

void AxialRectangularMap::initializeStoragePoints() {
    for (int r = 0; r < height; r++) {
        auto widthWithOffset = widthStorageOffset + width;
        for (int q = 0; q < widthWithOffset; q++) {
            // For simplicity initialize even the empty space.
            storage[r][q] = std::make_shared<Point>(q, r);
        }
    }
}

std::size_t AxialRectangularMap::getWidth() const {
    return width;
}

std::size_t AxialRectangularMap::getHeight() const {
    return height;
}

std::shared_ptr<Point> AxialRectangularMap::getPoint(int x, int y) {
    if (areCoordsOutOfBounds(x, y)) {
        throw std::out_of_range("Indices q=" + std::to_string(x) +
                                " r=" + std::to_string(y) + " are out of range.");
    }
    return storage[y][x];
}

std::vector<std::shared_ptr<Point>> AxialRectangularMap::getNeighbors(std::shared_ptr<Point> point) {
    int axialDirectionVectors[6][2] = {{1,  0},
                                       {0,  1},
                                       {-1, 1},
                                       {-1, 0},
                                       {0,  -1},
                                       {1,  -1}};

    std::vector<std::shared_ptr<Point>> neighbors;
    for (auto directionVector : axialDirectionVectors) {
        int newQ = directionVector[0] + point->getX();
        int newR = directionVector[1] + point->getY();
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
    if (q < widthStorageOffset - r / 2)
        return true;
    if (q >= width + widthStorageOffset - r / 2)
        return true;
    return false;
}

double AxialRectangularMap::euclideanDistance(std::shared_ptr<Point> lhs, std::shared_ptr<Point> rhs) {
    throw NotImplementedException();
}


