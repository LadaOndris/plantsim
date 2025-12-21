#pragma once

#include "simulation/GridTopology.h"
#include <cmath>

namespace initializers {

/**
 * @brief Amount policy that interpolates based on distance from the grid center.
 * 
 * @details Computes hex distance from the center of the topology and interpolates
 * between centerValue (at center) and edgeValue (at maximum distance).
 */
class DistanceFromCenter {
public:
    float centerValue;
    float edgeValue;

    constexpr DistanceFromCenter(float centerValue, float edgeValue)
        : centerValue(centerValue), edgeValue(edgeValue) {}

    [[nodiscard]] float compute(AxialCoord coord, const GridTopology& topology) const {
        OffsetCoord centerOffset{topology.width / 2, topology.height / 2};
        AxialCoord center = centerOffset.toAxialCoord();

        // Compute hex distance
        int dq = coord.q - center.q;
        int dr = coord.r - center.r;
        int ds = (-coord.q - coord.r) - (-center.q - center.r);
        int distance = (std::abs(dq) + std::abs(dr) + std::abs(ds)) / 2;

        // Maximum possible distance (approximate, from center to corner)
        int maxDistance = (topology.width + topology.height) / 2;

        // Normalize and interpolate
        float t = static_cast<float>(distance) / static_cast<float>(maxDistance);
        t = t > 1.0f ? 1.0f : t;

        return centerValue + t * (edgeValue - centerValue);
    }
};

} // namespace initializers
