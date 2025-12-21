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
    // Coordinate Helpers
    // =========================================================================

    /**
     * @brief Get the storage index for an offset coordinate (col, row).
     * 
     * Uses GridTopology's coordinate conversion.
     */
    [[nodiscard]] int index(int col, int row) const {
        AxialCoord axial = oddrToAxial({col, row});
        return topology.toIndex(axial);
    }

    [[nodiscard]] int index(OffsetCoord coord) const {
        return index(coord.col, coord.row);
    }

    /**
     * @brief Get the storage index for STORAGE coordinates (storageCol, storageRow).
     * 
     * Direct access to the underlying storage layout. Use this when testing
     * algorithms that operate on storage space (like LightComputation).
     */
    [[nodiscard]] int storageIndex(int storageCol, int storageRow) const {
        StorageCoord dim = topology.getStorageDimension();
        return storageRow * dim.x + storageCol;
    }

    /**
     * @brief Get storage dimensions.
     */
    [[nodiscard]] StorageCoord storageDim() const {
        return topology.getStorageDimension();
    }

    // =========================================================================
    // Offset Coordinate Access (for general use)
    // =========================================================================

    void setCellType(int col, int row, CellState::Type type) {
        state.cellTypes[index(col, row)] = static_cast<int>(type);
    }

    void setCellType(OffsetCoord coord, CellState::Type type) {
        setCellType(coord.col, coord.row, type);
    }

    [[nodiscard]] CellState::Type getCellType(int col, int row) const {
        return static_cast<CellState::Type>(state.cellTypes[index(col, row)]);
    }

    void setLight(int col, int row, float value) {
        state.light[index(col, row)] = value;
    }

    [[nodiscard]] float getLight(int col, int row) const {
        return state.light[index(col, row)];
    }

    [[nodiscard]] float getLight(OffsetCoord coord) const {
        return getLight(coord.col, coord.row);
    }

    void setPlantWater(int col, int row, float value) {
        state.plantWater[index(col, row)] = value;
    }

    [[nodiscard]] float getPlantWater(int col, int row) const {
        return state.plantWater[index(col, row)];
    }

    void setPlantSugar(int col, int row, float value) {
        state.plantSugar[index(col, row)] = value;
    }

    [[nodiscard]] float getPlantSugar(int col, int row) const {
        return state.plantSugar[index(col, row)];
    }

    [[nodiscard]] float getPlantSugar(OffsetCoord coord) const {
        return getPlantSugar(coord.col, coord.row);
    }

    // =========================================================================
    // Storage Coordinate Access (for light propagation tests)
    // =========================================================================

    void setCellTypeByStorage(int storageCol, int storageRow, CellState::Type type) {
        state.cellTypes[storageIndex(storageCol, storageRow)] = static_cast<int>(type);
    }

    [[nodiscard]] CellState::Type getCellTypeByStorage(int storageCol, int storageRow) const {
        return static_cast<CellState::Type>(state.cellTypes[storageIndex(storageCol, storageRow)]);
    }

    void setLightByStorage(int storageCol, int storageRow, float value) {
        state.light[storageIndex(storageCol, storageRow)] = value;
    }

    [[nodiscard]] float getLightByStorage(int storageCol, int storageRow) const {
        return state.light[storageIndex(storageCol, storageRow)];
    }

    void setPlantWaterByStorage(int storageCol, int storageRow, float value) {
        state.plantWater[storageIndex(storageCol, storageRow)] = value;
    }

    [[nodiscard]] float getPlantWaterByStorage(int storageCol, int storageRow) const {
        return state.plantWater[storageIndex(storageCol, storageRow)];
    }

    void setPlantSugarByStorage(int storageCol, int storageRow, float value) {
        state.plantSugar[storageIndex(storageCol, storageRow)] = value;
    }

    [[nodiscard]] float getPlantSugarByStorage(int storageCol, int storageRow) const {
        return state.plantSugar[storageIndex(storageCol, storageRow)];
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
            if (isValid(col, row)) {
                setCellType(col, row, type);
            }
        }
    }

    /**
     * @brief Fill an entire storage column with a cell type.
     */
    void fillStorageColumn(int storageCol, CellState::Type type) {
        for (int row = 0; row < topology.height; ++row) {
            setCellTypeByStorage(storageCol, row, type);
        }
    }

    /**
     * @brief Set plant water for a storage column.
     */
    void setWaterForStorageColumn(int storageCol, float value) {
        for (int row = 0; row < topology.height; ++row) {
            setPlantWaterByStorage(storageCol, row, value);
        }
    }

    /**
     * @brief Check if a coordinate is valid in the topology.
     */
    [[nodiscard]] bool isValid(int col, int row) const {
        AxialCoord axial = oddrToAxial({col, row});
        return topology.isValid(axial);
    }

    /**
     * @brief Check if storage coordinates are valid.
     */
    [[nodiscard]] bool isStorageValid(int storageCol, int storageRow) const {
        return topology.isValid(StorageCoord{storageCol, storageRow});
    }

    /**
     * @brief Iterate over all valid cells calling the callback.
     */
    void forEachValidCell(std::function<void(int col, int row)> callback) const {
        for (int row = 0; row < topology.height; ++row) {
            for (int col = 0; col < topology.width; ++col) {
                if (isValid(col, row)) {
                    callback(col, row);
                }
            }
        }
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
        return storageDim().x;
    }

    /**
     * @brief Get storage height (same as logical height).
     */
    [[nodiscard]] int storageHeight() const {
        return storageDim().y;
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
