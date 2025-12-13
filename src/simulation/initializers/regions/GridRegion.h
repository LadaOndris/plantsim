#pragma once

#include "simulation/GridTopology.h"

namespace initializers {

/**
 * @brief Region selector for a rectangular subgrid within the topology.
 * 
 * @details Defines a rectangular region starting at (offsetX, offsetY) in offset coordinates,
 * with dimensions width x height. Coordinates are converted to axial for containment checks.
 */
class GridRegion {
public:
    int width;
    int height;
    int offsetX;
    int offsetY;

    /**
     * @brief Construct a grid region using offset coordinates.
     * 
     * @param width Width of the region in cells
     * @param height Height of the region in cells
     * @param offsetX X offset (column) in offset coordinates
     * @param offsetY Y offset (row) in offset coordinates
     */
    constexpr GridRegion(int width, int height, int offsetX = 0, int offsetY = 0)
        : width(width), height(height), offsetX(offsetX), offsetY(offsetY) {}

    /**
     * @brief Check if an axial coordinate is within this region.
     */
    [[nodiscard]] bool contains(AxialCoord coord, const GridTopology& topology) const {
        if (!topology.isValid(coord)) {
            return false;
        }

        // Convert axial to offset coordinates for comparison
        OffsetCoord offset = axialToOddr(coord);
        
        return offset.col >= offsetX && offset.col < offsetX + width &&
               offset.row >= offsetY && offset.row < offsetY + height;
    }
};

} // namespace initializers
