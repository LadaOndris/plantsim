#pragma once

#include "simulation/GridTopology.h"
#include "simulation/State.h"
#include "simulation/CellState.h"

namespace initializers {

/**
 * @brief Action policy that sets the cell type for cells.
 */
class SetCellType {
public:
    explicit constexpr SetCellType(CellState::Type type) : cellType(type) {}

    void apply(AxialCoord coord, const GridTopology& topology, State& state) const {
        OffsetCoord offset = axialToOddr(coord);
        int index = offset.row * topology.width + offset.col;
        state.cellTypes[index] = static_cast<int>(cellType);
    }
private:
    CellState::Type cellType;
};

} // namespace initializers
