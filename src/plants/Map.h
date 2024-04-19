//
// Created by lada on 1/31/22.
//

#ifndef PLANTSIM_MAP_H
#define PLANTSIM_MAP_H

#include <memory>
#include <vector>
#include "Point.h"
#include "plants/iterators/Iterator.h"

class Map {
public:
    virtual ~Map() = default;

    [[nodiscard]] virtual std::size_t getWidth() const = 0;

    [[nodiscard]] virtual std::size_t getHeight() const = 0;

    virtual Point *getPoint(int x, int y) = 0;

    [[nodiscard]] virtual std::vector<Point *> &getPoints() = 0;

    [[nodiscard]] virtual std::vector<Point *> getNeighbors(const Point &point) = 0;

    virtual double euclideanDistance(const Point &lhs, const Point &rhs) const = 0;
};


#endif //PLANTSIM_MAP_H
