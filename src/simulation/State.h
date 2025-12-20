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
    
    /// Resources for each cell (size = width * height)
    std::vector<float> resources;
    
    /// Cell types for each cell (size = width * height), values from CellState::Type
    std::vector<int> cellTypes;
    
    /// Soil water concentration for each cell (size = width * height)
    std::vector<float> soilWater;
    
    /// Soil mineral concentration for each cell (size = width * height)
    std::vector<float> soilMineral;
    
    State() = default;

    State(int width, int height, 
          std::vector<float> resources, 
          std::vector<int> cellTypes)
        : width(width)
        , height(height)
        , resources(std::move(resources))
        , cellTypes(std::move(cellTypes))
        , soilWater(this->resources.size(), 0.0f)
        , soilMineral(this->resources.size(), 0.0f)
    {}
    
    State(int width, int height, 
          std::vector<float> resources, 
          std::vector<int> cellTypes,
          std::vector<float> soilWater,
          std::vector<float> soilMineral)
        : width(width)
        , height(height)
        , resources(std::move(resources))
        , cellTypes(std::move(cellTypes))
        , soilWater(std::move(soilWater))
        , soilMineral(std::move(soilMineral))
    {}

    /**
     * @brief Returns the total number of cells in the grid.
     */
    [[nodiscard]] size_t totalCells() const {
        return static_cast<size_t>(width) * height;
    }
};

using StatePtr = std::shared_ptr<State>;
