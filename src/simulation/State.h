#pragma once

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
    /// Width of the simulation grid
    int width = 0;
    
    /// Height of the simulation grid
    int height = 0;
    
    /// Cell types for each cell (size = width * height), values from CellState::Type
    std::vector<int> cellTypes;
    
    // ========== Soil fields (environmental) ==========
    /// Soil water concentration for each cell (size = width * height)
    std::vector<float> soilWater;
    
    /// Soil mineral concentration for each cell (size = width * height)
    std::vector<float> soilMineral;
    
    // ========== Plant internal stores ==========
    /// Plant sugar (energy) for each cell (size = width * height)
    std::vector<float> plantSugar;
    
    /// Plant water for each cell (size = width * height)
    std::vector<float> plantWater;
    
    /// Plant mineral for each cell (size = width * height)
    std::vector<float> plantMineral;
    
    State() = default;

    State(int width, int height, 
          std::vector<int> cellTypes)
        : width(width)
        , height(height)
        , cellTypes(std::move(cellTypes))
        , soilWater(this->cellTypes.size(), 0.0f)
        , soilMineral(this->cellTypes.size(), 0.0f)
        , plantSugar(this->cellTypes.size(), 0.0f)
        , plantWater(this->cellTypes.size(), 0.0f)
        , plantMineral(this->cellTypes.size(), 0.0f)
    {}
    
    State(int width, int height, 
          std::vector<int> cellTypes,
          std::vector<float> soilWater,
          std::vector<float> soilMineral,
          std::vector<float> plantSugar,
          std::vector<float> plantWater,
          std::vector<float> plantMineral)
        : width(width)
        , height(height)
        , cellTypes(std::move(cellTypes))
        , soilWater(std::move(soilWater))
        , soilMineral(std::move(soilMineral))
        , plantSugar(std::move(plantSugar))
        , plantWater(std::move(plantWater))
        , plantMineral(std::move(plantMineral))
    {}

    /**
     * @brief Returns the total number of cells in the grid.
     */
    [[nodiscard]] size_t totalCells() const {
        return static_cast<size_t>(width) * height;
    }
};

using StatePtr = std::shared_ptr<State>;
