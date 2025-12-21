#pragma once

#include "simulation/GridTopology.h"

namespace initializers {

/**
 * @brief Axis specification for gradient calculations.
 */
enum class GradientAxis {
    Q,  ///< Gradient along the Q axis (axial)
    R   ///< Gradient along the R axis (axial)
};

/**
 * @brief Amount policy that computes a linear gradient across the grid.
 * 
 * @details Interpolates between startValue and endValue based on position
 * along the specified axis relative to the topology dimensions.
 */
class LinearGradient {
public:
    float startValue;
    float endValue;
    GradientAxis axis;

    constexpr LinearGradient(float startValue, float endValue, GradientAxis axis = GradientAxis::Q)
        : startValue(startValue), endValue(endValue), axis(axis) {}

    [[nodiscard]] float compute(AxialCoord coord, const GridTopology& topology) const {
        float t;
        if (axis == GradientAxis::Q) {
            // Normalize q position to [0, 1]
            // For valid coords, q ranges roughly from -height/2 to width + height/2
            // Use offset coords for cleaner normalization
            OffsetCoord offset = coord.toOffsetCoord();
            t = static_cast<float>(offset.col) / static_cast<float>(topology.width - 1);
        } else {
            // Normalize r position to [0, 1]
            t = static_cast<float>(coord.r) / static_cast<float>(topology.height - 1);
        }

        // Clamp t to [0, 1]
        t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);

        return startValue + t * (endValue - startValue);
    }
};

} // namespace initializers
