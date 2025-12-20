#pragma once

#include "simulation/GridTopology.h"
#include "simulation/State.h"
#include <tuple>
#include <utility>

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

    void apply(AxialCoord coord, const GridTopology& topology, State& state) const {
        applyImpl(coord, topology, state, std::index_sequence_for<Actions...>{});
    }

private:
    template<std::size_t... Is>
    void applyImpl(AxialCoord coord, const GridTopology& topology,
                   State& state,
                   std::index_sequence<Is...>) const {
        (std::get<Is>(actions).apply(coord, topology, state), ...);
    }
};

template<typename... Actions>
CompositeAction(Actions...) -> CompositeAction<Actions...>;

} // namespace initializers
