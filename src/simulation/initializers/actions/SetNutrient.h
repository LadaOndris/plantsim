#pragma once

#include "simulation/GridTopology.h"
#include <vector>

namespace initializers {

/**
 * @brief Action policy that sets nutrient values for cells.
 * 
 * @tparam AmountPolicy A policy type that provides float compute(AxialCoord, GridTopology&)
 */
template<typename AmountPolicy>
class SetNutrient {
public:
    explicit SetNutrient(AmountPolicy amount) : amountPolicy(std::move(amount)) {}

    void apply(AxialCoord coord, const GridTopology& topology,
               std::vector<float>& resources, std::vector<int>& cellTypes,
               std::vector<float>& nutrients) const {
        // Convert axial to offset coordinates for flat indexing
        OffsetCoord offset = axialToOddr(coord);
        int index = offset.row * topology.width + offset.col;
        nutrients[index] = amountPolicy.compute(coord, topology);
    }
    
private:
    AmountPolicy amountPolicy;
};

template<typename AmountPolicy>
SetNutrient(AmountPolicy) -> SetNutrient<AmountPolicy>;

} // namespace initializers
