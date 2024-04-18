//
// Created by lada on 8/26/21.
//

#ifndef PLANTSIM_POINT_H
#define PLANTSIM_POINT_H

#include <utility>

struct Point {
    enum Type {
        Air,
        Cell
    };

    explicit Point(int x, int y);

    std::pair<int, int> coords;
    int resources;
    Type type;

    [[nodiscard]] int getX() const {
        return coords.first;
    }

    [[nodiscard]] int getY() const {
        return coords.second;
    }

    bool operator==(const Point &other) const {
        return coords == other.coords;
    }
};


#endif //PLANTSIM_POINT_H
