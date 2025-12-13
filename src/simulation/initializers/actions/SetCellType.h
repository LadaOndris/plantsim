#pragma once

#include "simulation/GridTopology.h"
#include "simulation/CellState.h"
#include <vector>

namespace initializers {

/**
 * @brief Action policy that sets the cell type for cells.
 */
class SetCellType {
public:
    explicit constexpr SetCellType(CellState::Type type) : cellType(type) {}

    void apply(AxialCoord coord, const GridTopology& topology,
               std::vector<float>& resources, std::vector<int>& cellTypes) const {
        // Convert axial to offset coordinates for flat indexing
        OffsetCoord offset = axialToOddr(coord);
        int index = offset.row * topology.width + offset.col;
        cellTypes[index] = static_cast<int>(cellType);
    }
private:
    CellState::Type cellType;
};

} // namespace initializers
