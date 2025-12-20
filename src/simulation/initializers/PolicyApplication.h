#pragma once

#include "simulation/GridTopology.h"
#include <utility>
#include <vector>

namespace initializers {

/**
 * @brief Combines a region policy with an action policy.
 * 
 * @details When applied, iterates over all cells in the topology,
 * checks if each cell is within the region, and if so, applies the action.
 * 
 * @tparam RegionPolicy A policy type with bool contains(AxialCoord, GridTopology&) const
 * @tparam ActionPolicy A policy type with void apply(AxialCoord, GridTopology&, vector<float>&, vector<int>&) const
 */
template<typename RegionPolicy, typename ActionPolicy>
class PolicyApplication {
public:
    RegionPolicy region;
    ActionPolicy action;

    PolicyApplication(RegionPolicy region, ActionPolicy action)
        : region(std::move(region)), action(std::move(action)) {}

    /**
     * @brief Apply the action to all cells in the region.
     * 
     * @param topology The grid topology
     * @param resources Resource vector (size = topology.totalCells())
     * @param cellTypes Cell type vector (size = topology.totalCells())
     * @param nutrients Nutrient vector (size = topology.totalCells())
     */
    void apply(const GridTopology& topology,
               std::vector<float>& resources,
               std::vector<int>& cellTypes,
               std::vector<float>& nutrients) const {
        // Iterate over all logical cells using offset coordinates
        for (int row = 0; row < topology.height; ++row) {
            for (int col = 0; col < topology.width; ++col) {
                AxialCoord axial = oddrToAxial({col, row});
                
                if (region.contains(axial, topology)) {
                    action.apply(axial, topology, resources, cellTypes, nutrients);
                }
            }
        }
    }
};

template<typename RegionPolicy, typename ActionPolicy>
PolicyApplication(RegionPolicy, ActionPolicy) -> PolicyApplication<RegionPolicy, ActionPolicy>;

} // namespace initializers
