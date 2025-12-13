#pragma once

#include <memory>
#include <vector>
#include "Point.h"
#include "plants/iterators/Iterator.h"

class Map {
public:
    virtual ~Map() = default;

    [[nodiscard]] virtual int getWidth() const = 0;

    [[nodiscard]] virtual int getHeight() const = 0;

    [[nodiscard]] virtual std::pair<int, int> getMaxCoords() const = 0;

    virtual Point *getPoint(int x, int y) = 0;

    [[nodiscard]] virtual std::vector<Point *> &getPoints() = 0;

    [[nodiscard]] virtual std::vector<Point *> getNeighbors(const Point &point) = 0;

    [[nodiscard]] virtual const std::vector<std::pair<int, int>> &getNeighborOffsets() const = 0;

    virtual double euclideanDistance(const Point &lhs, const Point &rhs) const = 0;
};


