#pragma once

#include "simulation/CellState.h"
#include "simulation/GridTopology.h"
#include <vector>
#include <utility>
#include <memory>

/**
 * @brief Represents the state of the simulation at a given time step.
 * 
 * This class holds all the data needed to describe the simulation state,
 * independent of the simulator implementation (CPU, SYCL, CUDA).
 */
class State {
public:
    int width = 0;
    int height = 0;
    
    StorageCoord storageDim;

    std::vector<int>   cellTypes;

    std::vector<float> soilWater;
    std::vector<float> soilMineral;

    std::vector<float> light;

    std::vector<float> plantSugar;
    std::vector<float> plantWater;
    std::vector<float> plantMineral;
    
    explicit State(const GridTopology& topology,
                   float defaultSoilWater = 0.0f,
                   float defaultSoilMineral = 0.0f)
        : width(topology.width)
        , height(topology.height)
        , storageDim(topology.storageDim)
    {
        const size_t n = static_cast<size_t>(storageDim.x) * static_cast<size_t>(storageDim.y);

        cellTypes.assign(n, static_cast<int>(CellState::Padding));
        initializeCellsToAir(topology);

        soilWater.assign(n, defaultSoilWater);
        soilMineral.assign(n, defaultSoilMineral);

        light.assign(n, 0.0f);

        plantSugar.assign(n, 0.0f);
        plantWater.assign(n, 0.0f);
        plantMineral.assign(n, 0.0f);
    }

    void initializeCellsToAir(const GridTopology& topology) {
        for (int row = 0; row < topology.height; ++row) {
            for (int col = 0; col < topology.width; ++col) {
                const int idx = topology.toStorageIndex(OffsetCoord{col, row});
                cellTypes[idx] = static_cast<int>(CellState::Air);
            }
        }
    }

    [[nodiscard]] size_t totalLogicalCells() const {
        return static_cast<size_t>(width) * static_cast<size_t>(height);
    }

    [[nodiscard]] size_t totalStorageCells() const {
        return static_cast<size_t>(storageDim.x) * static_cast<size_t>(storageDim.y);
    }
};

using StatePtr = std::shared_ptr<State>;
