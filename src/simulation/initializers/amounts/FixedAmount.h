#pragma once

#include "simulation/GridTopology.h"

namespace initializers {

/**
 * @brief Amount policy that returns a fixed runtime value.
 */
class FixedAmount {
public:
    constexpr explicit FixedAmount(float value) : value(value) {}

    [[nodiscard]] constexpr float compute(AxialCoord coord, const GridTopology& topology) const {
        return value;
    }
private:
    float value;
};

} // namespace initializers
