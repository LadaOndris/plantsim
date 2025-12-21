#pragma once

#include "simulation/GridTopology.h"

namespace initializers {

/**
 * @brief Region selector for a single cell.
 */
class SingleCell {
public:
    /**
     * @brief Construct from axial coordinates.
     */
    constexpr explicit SingleCell(AxialCoord coord) : target(coord) {}

    /**
     * @brief Construct from offset coordinates.
     */
    explicit SingleCell(OffsetCoord coord) : target(coord.toAxialCoord()) {}

    [[nodiscard]] constexpr bool contains(AxialCoord coord, const GridTopology& topology) const {
        return coord == target && topology.isValid(coord);
    }
private:
    AxialCoord target;
};

} // namespace initializers
