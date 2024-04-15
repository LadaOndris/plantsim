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

    virtual std::size_t getWidth() const = 0;
    virtual std::size_t getHeight() const = 0;
    virtual std::shared_ptr<Point> getPoint(int x, int y) = 0;
    virtual std::vector<std::shared_ptr<Point>> getPoints() const = 0;
    virtual std::vector<std::shared_ptr<Point>> getNeighbors(std::shared_ptr<Point> point) = 0;
    virtual double euclideanDistance(std::shared_ptr<Point> lhs, std::shared_ptr<Point> rhs) = 0;
};


#endif //PLANTSIM_MAP_H
