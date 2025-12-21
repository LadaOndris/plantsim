#pragma once

#include "simulation/GridTopology.h"
#include "simulation/State.h"
#include <Eigen/Dense>
#include <utility>

namespace initializers {

/**
 * @brief Combines a region policy with an action policy.
 * 
 * @details When applied, iterates over all cells in the topology,
 * checks if each cell is within the region, and if so, applies the action.
 * 
 * @tparam RegionPolicy A policy type with bool contains(AxialCoord, GridTopology&) const
 * @tparam ActionPolicy A policy type with void apply(AxialCoord, GridTopology&, vector<float>&, vector<int>&) const
 *                      or void apply(AxialCoord, GridTopology&, Eigen::MatrixXf&) const
 */
template<typename RegionPolicy, typename ActionPolicy>
class PolicyApplication {
public:
    using MatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    RegionPolicy region;
    ActionPolicy action;

    PolicyApplication(RegionPolicy region, ActionPolicy action)
        : region(std::move(region)), action(std::move(action)) {}

    /**
     * @brief Apply the action to all cells in the region.
     * 
     * @param topology The grid topology
     * @param state The state object containing all fields (flat/logical layout)
     */
    void apply(const GridTopology& topology, State& state) const {
        // Iterate over all logical cells using offset coordinates
        for (int row = 0; row < topology.height; ++row) {
            for (int col = 0; col < topology.width; ++col) {
                OffsetCoord offset{col, row};
                AxialCoord axial = offset.toAxialCoord();
                
                if (region.contains(axial, topology)) {
                    action.apply(axial, topology, state);
                }
            }
        }
    }

    /**
     * @brief Apply the action to all cells in the region (Eigen matrix version).
     * 
     * @param topology The grid topology
     * @param matrix The Eigen matrix to modify (storage layout: row-major, size = height x width)
     */
    void apply(const GridTopology& topology, MatrixXf& matrix) const {
        // Iterate over all logical cells using offset coordinates
        for (int row = 0; row < topology.height; ++row) {
            for (int col = 0; col < topology.width; ++col) {
                OffsetCoord offset{col, row};
                AxialCoord axial = offset.toAxialCoord();
                
                if (region.contains(axial, topology)) {
                    action.apply(axial, topology, matrix);
                }
            }
        }
    }
};

template<typename RegionPolicy, typename ActionPolicy>
PolicyApplication(RegionPolicy, ActionPolicy) -> PolicyApplication<RegionPolicy, ActionPolicy>;

} // namespace initializers
