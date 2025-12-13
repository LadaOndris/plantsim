#pragma once

#include <utility>

/**
 * Represents a point in the axial coordinate system.
 * Contains only coordinate information.
 * Cell-specific data (type, resources) is stored in CellState.
 */
struct Point {
    explicit Point(int x, int y);

    std::pair<int, int> coords;

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


