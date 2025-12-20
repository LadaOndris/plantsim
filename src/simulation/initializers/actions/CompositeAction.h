#pragma once

#include "simulation/GridTopology.h"
#include <tuple>
#include <utility>
#include <vector>

namespace initializers {

/**
 * @brief Composite action that applies multiple actions in sequence.
 * 
 * @tparam Actions The action policy types to apply
 */
template<typename... Actions>
class CompositeAction {
public:
    std::tuple<Actions...> actions;

    explicit CompositeAction(Actions... actions)
        : actions(std::move(actions)...) {}

    void apply(AxialCoord coord, const GridTopology& topology,
               std::vector<float>& resources, std::vector<int>& cellTypes,
               std::vector<float>& nutrients) const {
        applyImpl(coord, topology, resources, cellTypes, nutrients, std::index_sequence_for<Actions...>{});
    }

private:
    template<std::size_t... Is>
    void applyImpl(AxialCoord coord, const GridTopology& topology,
                   std::vector<float>& resources, std::vector<int>& cellTypes,
                   std::vector<float>& nutrients,
                   std::index_sequence<Is...>) const {
        (std::get<Is>(actions).apply(coord, topology, resources, cellTypes, nutrients), ...);
    }
};

template<typename... Actions>
CompositeAction(Actions...) -> CompositeAction<Actions...>;

} // namespace initializers
