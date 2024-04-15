//
// Created by lada on 8/26/21.
//

#include "Point.h"

Point::~Point() = default;

Point::Point(int x, int y)
        : coords(x, y) {

}

int Point::getX() const {
    return coords.first;
}

int Point::getY() const {
    return coords.second;
}

std::pair<int, int> Point::getCoords() const {
    return coords;
}
