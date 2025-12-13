#pragma once

#include "simulation/GridTopology.h"
#include <tuple>
#include <utility>

namespace initializers {

/**
 * @brief Composite region that combines multiple regions using union logic.
 * 
 * @details A cell is considered within this region if it belongs to any of the
 * constituent regions.
 * 
 * @tparam Regions The region policy types to combine
 */
template<typename... Regions>
class CompositeRegion {
public:
    std::tuple<Regions...> regions;

    constexpr explicit CompositeRegion(Regions... regions)
        : regions(std::move(regions)...) {}

    [[nodiscard]] bool contains(AxialCoord coord, const GridTopology& topology) const {
        return containsImpl(coord, topology, std::index_sequence_for<Regions...>{});
    }

private:
    template<std::size_t... Is>
    [[nodiscard]] bool containsImpl(AxialCoord coord, const GridTopology& topology,
                                     std::index_sequence<Is...>) const {
        return (std::get<Is>(regions).contains(coord, topology) || ...);
    }
};

template<typename... Regions>
CompositeRegion(Regions...) -> CompositeRegion<Regions...>;

} // namespace initializers
