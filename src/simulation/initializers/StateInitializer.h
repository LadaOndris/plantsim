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

        // Create a temporary state with flat (logical) layout for initialization
        State flatState;
        flatState.width = topology.width;
        flatState.height = topology.height;
        flatState.resources.resize(totalCells, 0.0f);
        flatState.cellTypes.resize(totalCells, static_cast<int>(CellState::Air));
        flatState.soilWater.resize(totalCells, 0.0f);
        flatState.soilMineral.resize(totalCells, 0.0f);

        applyPolicies(topology, flatState, std::index_sequence_for<Policies...>{});

        // Convert to storage layout
        auto storedResources = store<float>(flatState.resources, topology.width, topology.height, -1.0f);
        auto storedCellTypes = store<int>(flatState.cellTypes, topology.width, topology.height, -1);
        auto storedSoilWater = store<float>(flatState.soilWater, topology.width, topology.height, 0.0f);
        auto storedSoilMineral = store<float>(flatState.soilMineral, topology.width, topology.height, 0.0f);

        return State(topology.width, topology.height, 
                     std::move(storedResources), 
                     std::move(storedCellTypes),
                     std::move(storedSoilWater),
                     std::move(storedSoilMineral));
    }

private:
    std::tuple<Policies...> policies;

    template<std::size_t... Is>
    void applyPolicies(const GridTopology& topology,
                       State& state,
                       std::index_sequence<Is...>) const {
        (std::get<Is>(policies).apply(topology, state), ...);
    }
};

template<typename... Policies>
StateInitializer(Policies...) -> StateInitializer<Policies...>;

} // namespace initializers
