#pragma once

#include <vector>
#include <utility>

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
    std::vector<int> resources;
    
    /// Cell types for each cell (size = width * height), values from CellState::Type
    std::vector<int> cellTypes;
    
    /// Neighbor offsets defining the neighborhood relationship (e.g., for hex grid)
    std::vector<std::pair<int, int>> neighborOffsets;

    State() = default;

    State(int width, int height, 
          std::vector<int> resources, 
          std::vector<int> cellTypes,
          std::vector<std::pair<int, int>> neighborOffsets)
        : width(width)
        , height(height)
        , resources(std::move(resources))
        , cellTypes(std::move(cellTypes))
        , neighborOffsets(std::move(neighborOffsets)) 
    {}

    /**
     * @brief Returns the total number of cells in the grid.
     */
    [[nodiscard]] size_t totalCells() const {
        return static_cast<size_t>(width) * height;
    }

    /**
     * @brief Converts 2D coordinates to linear index.
     */
    [[nodiscard]] int toIndex(int r, int q) const {
        return r * width + q;
    }

    /**
     * @brief Checks if coordinates are within bounds.
     */
    [[nodiscard]] bool isValid(int r, int q) const {
        return r >= 0 && r < height && q >= 0 && q < width;
    }
};