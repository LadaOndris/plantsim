#pragma once

#include "simulation/GridTopology.h"
#include "simulation/State.h"
#include "simulation/Options.h"
#include "simulation/CellState.h"
#include "simulation/initializers/Initializers.h"

#include <functional>

class SimulationTestHelper {
public:
    GridTopology topology;
    State state;
    Options options;

    SimulationTestHelper(int width, int height)
        : topology(width, height)
        , state(createEmptyState(topology))
        , options{}
    {
    }

    // =========================================================================
    // Offset Coordinate Access (for general use)
    // =========================================================================

    void setCellType(OffsetCoord coord, CellState::Type type) {
        int index = topology.toStorageIndex(coord);
        state.cellTypes[index] = static_cast<int>(type);
    }

    [[nodiscard]] CellState::Type getCellType(OffsetCoord coord) const {
        return static_cast<CellState::Type>(state.cellTypes[topology.toStorageIndex(coord)]);
    }

    void setLight(OffsetCoord coord, float value) {
        state.light[topology.toStorageIndex(coord)] = value;
    }

    [[nodiscard]] float getLight(OffsetCoord coord) const {
        return state.light[topology.toStorageIndex(coord)];
    }

    void setPlantWater(OffsetCoord coord, float value) {
        state.plantWater[topology.toStorageIndex(coord)] = value;
    }

    [[nodiscard]] float getPlantWater(OffsetCoord coord) const {
        return state.plantWater[topology.toStorageIndex(coord)];
    }

    void setPlantSugar(OffsetCoord coord, float value) {
        state.plantSugar[topology.toStorageIndex(coord)] = value;
    }

    [[nodiscard]] float getPlantSugar(OffsetCoord coord) const {
        return state.plantSugar[topology.toStorageIndex(coord)];
    }

    // =========================================================================
    // Convenience Methods
    // =========================================================================

    /**
     * @brief Fill an entire offset column with a cell type.
     * Light propagates through offset columns in the logical grid.
     */
    void fillColumn(int col, CellState::Type type) {
        for (int row = 0; row < topology.height; ++row) {
            OffsetCoord coord{col, row};
            if (isValid(coord)) {
                setCellType(coord, type);
            }
        }
    }

    /**
     * @brief Check if a coordinate is valid in the topology.
     */
    [[nodiscard]] bool isValid(OffsetCoord coord) const {
        AxialCoord axial = coord.toAxialCoord();
        return topology.isValid(axial);
    }

    /**
     * @brief Get topmost row index (where light enters from sky).
     */
    [[nodiscard]] int topRow() const {
        return topology.height - 1;
    }

    /**
     * @brief Get bottommost row index (ground level).
     */
    [[nodiscard]] int bottomRow() const {
        return 0;
    }

    /**
     * @brief Get storage width.
     */
    [[nodiscard]] int storageWidth() const {
        return topology.storageDim.x;
    }

    /**
     * @brief Get storage height (same as logical height).
     */
    [[nodiscard]] int storageHeight() const {
        return topology.storageDim.y;
    }

private:
    static State createEmptyState(const GridTopology& topo) {
        using namespace initializers;
        StateInitializer initializer{
            PolicyApplication{FullGrid{}, SetCellType{CellState::Type::Air}}
        };
        return initializer.initialize(topo);
    }
};
