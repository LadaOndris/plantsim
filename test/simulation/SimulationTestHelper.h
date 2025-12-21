#pragma once

#include "simulation/GridTopology.h"
#include "simulation/State.h"
#include "simulation/Options.h"
#include "simulation/CellState.h"
#include "simulation/initializers/Initializers.h"


class SimulationTestHelper {
public:
    GridTopology topology;
    State state;
    Options options;

    SimulationTestHelper(const GridTopology &topology)
        : topology(topology)
        , state(createEmptyState(topology))
        , options{}
    {
    }

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

    void fillColumn(int col, CellState::Type type) {
        for (int row = 0; row < topology.height; ++row) {
            OffsetCoord coord{col, row};
            if (topology.isValid(coord)) {
                setCellType(coord, type);
            }
        }
    }

private:
    static State createEmptyState(const GridTopology& topology) {
        using namespace initializers;
        StateInitializer initializer{
            PolicyApplication{FullGrid{}, SetCellType{CellState::Type::Air}}
        };
        return initializer.initialize(topology);
    }
};
