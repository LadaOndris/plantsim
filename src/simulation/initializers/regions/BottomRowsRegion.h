#pragma once

#include "simulation/GridTopology.h"

namespace initializers {

/**
 * @brief Region selector for the bottom N rows of the grid (soil layer).
 * 
 * @details In offset coordinates, the bottom rows have the highest row indices.
 * This is useful for creating soil layers that span the full width.
 */
class BottomRowsRegion {
public:
    int rowCount;

    /**
     * @brief Construct a region covering the bottom N rows.
     * 
     * @param rowCount Number of bottom rows to include
     */
    constexpr explicit BottomRowsRegion(int rowCount)
        : rowCount(rowCount) {}

    /**
     * @brief Check if an axial coordinate is within the bottom rows.
     */
    [[nodiscard]] bool contains(AxialCoord coord, const GridTopology& topology) const {
        if (!topology.isValid(coord)) {
            return false;
        }

        OffsetCoord offset = axialToOddr(coord);

        return offset.row < rowCount;
    }
};

} // namespace initializers
