#pragma once

#include "simulation/State.h"
#include "simulation/GridTopology.h"
#include "simulation/CellState.h"
#include <tuple>
#include <utility>
#include <vector>

namespace initializers {

/**
 * @brief Template-based state initializer that applies policies in sequence.
 * 
 * @details Each policy is a PolicyApplication that pairs a region with an action.
 * Policies are applied in order, so later policies can overwrite earlier ones.
 * 
 * @tparam Policies PolicyApplication types to apply
 * 
 * @example
 * ```cpp
 * StateInitializer initializer{
 *     PolicyApplication{FullGrid{}, SetCellType{CellState::Air}},
 *     PolicyApplication{GridRegion{10, 10, 5, 5}, SetCellType{CellState::Cell}},
 *     PolicyApplication{SingleCell{AxialCoord{5, 5}}, SetResource{FixedAmount{10000.0f}}}
 * };
 * State state = initializer.initialize(topology);
 * ```
 */
template<typename... Policies>
class StateInitializer {
public:
    explicit StateInitializer(Policies... policies)
        : policies(std::move(policies)...) {}

    /**
     * @brief Initialize the state by applying all policies to the topology.
     * 
     * @param topology The grid topology defining the simulation space
     * @return State The initialized state with resources and cell types set
     */
    [[nodiscard]] State initialize(const GridTopology& topology) const {
        const size_t totalCells = topology.totalCells();

        // Initialize with default values
        std::vector<float> resources(totalCells, 0.0f);
        std::vector<int> cellTypes(totalCells, static_cast<int>(CellState::Air));

        // Apply all policies in order
        applyPolicies(topology, resources, cellTypes, std::index_sequence_for<Policies...>{});

        // Convert to storage layout
        auto storedResources = store<float>(resources, topology.width, topology.height, -1.0f);
        auto storedCellTypes = store<int>(cellTypes, topology.width, topology.height, -1);

        return State(topology.width, topology.height, 
                     std::move(storedResources), 
                     std::move(storedCellTypes));
    }

private:
    std::tuple<Policies...> policies;

    template<std::size_t... Is>
    void applyPolicies(const GridTopology& topology,
                       std::vector<float>& resources,
                       std::vector<int>& cellTypes,
                       std::index_sequence<Is...>) const {
        (std::get<Is>(policies).apply(topology, resources, cellTypes), ...);
    }
};

template<typename... Policies>
StateInitializer(Policies...) -> StateInitializer<Policies...>;

} // namespace initializers
