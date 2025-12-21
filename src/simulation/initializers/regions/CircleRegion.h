#pragma once

#include "simulation/GridTopology.h"
#include <cmath>

namespace initializers {

/**
 * @brief Region selector for a circular area around a center point.
 * 
 * @details Uses hex distance (number of hex steps) for radius calculation.
 */
class CircleRegion {
public:
    /**
     * @brief Construct from axial coordinates and radius.
     */
    constexpr CircleRegion(AxialCoord center, int radius)
        : center(center), radius(radius) {}

    /**
     * @brief Construct from offset coordinates and radius.
     */
    CircleRegion(OffsetCoord center, int radius)
        : center(center.toAxialCoord()), radius(radius) {}

    [[nodiscard]] bool contains(AxialCoord coord, const GridTopology& topology) const {
        if (!topology.isValid(coord)) {
            return false;
        }

        int distance = axialDistance(coord, center);
        return distance <= radius;
    }

    float axialDistance(AxialCoord a, AxialCoord b) const {
        return (std::abs(a.q - b.q) + 
                std::abs(a.q + a.r - b.q - b.r) + 
                std::abs(a.r - b.r)) / 2.f;
    }
private:
    AxialCoord center;
    int radius;
};

} // namespace initializers
