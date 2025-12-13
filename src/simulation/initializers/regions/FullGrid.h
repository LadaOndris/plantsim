#pragma once

#include "simulation/GridTopology.h"

namespace initializers {

/**
 * @brief Region selector that includes all valid cells in the topology.
 */
class FullGrid {
public:
    constexpr FullGrid() = default;

    [[nodiscard]] constexpr bool contains(AxialCoord coord, const GridTopology& topology) const {
        return topology.isValid(coord);
    }
};

} // namespace initializers
