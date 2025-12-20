#pragma once

#include "simulation/GridTopology.h"
#include <vector>

namespace initializers {

/**
 * @brief Action policy that sets soil water values for cells.
 * 
 * @tparam AmountPolicy A policy type that provides float compute(AxialCoord, GridTopology&)
 */
template<typename AmountPolicy>
class SetSoilWater {
public:
    explicit SetSoilWater(AmountPolicy amount) : amountPolicy(std::move(amount)) {}

    void apply(AxialCoord coord, const GridTopology& topology,
               std::vector<float>& resources, std::vector<int>& cellTypes,
               std::vector<float>& soilWater, std::vector<float>& soilMineral) const {
        // Convert axial to offset coordinates for flat indexing
        OffsetCoord offset = axialToOddr(coord);
        int index = offset.row * topology.width + offset.col;
        soilWater[index] = amountPolicy.compute(coord, topology);
    }
    
private:
    AmountPolicy amountPolicy;
};

template<typename AmountPolicy>
SetSoilWater(AmountPolicy) -> SetSoilWater<AmountPolicy>;

/**
 * @brief Action policy that sets soil mineral values for cells.
 * 
 * @tparam AmountPolicy A policy type that provides float compute(AxialCoord, GridTopology&)
 */
template<typename AmountPolicy>
class SetSoilMineral {
public:
    explicit SetSoilMineral(AmountPolicy amount) : amountPolicy(std::move(amount)) {}

    void apply(AxialCoord coord, const GridTopology& topology,
               std::vector<float>& resources, std::vector<int>& cellTypes,
               std::vector<float>& soilWater, std::vector<float>& soilMineral) const {
        // Convert axial to offset coordinates for flat indexing
        OffsetCoord offset = axialToOddr(coord);
        int index = offset.row * topology.width + offset.col;
        soilMineral[index] = amountPolicy.compute(coord, topology);
    }
    
private:
    AmountPolicy amountPolicy;
};

template<typename AmountPolicy>
SetSoilMineral(AmountPolicy) -> SetSoilMineral<AmountPolicy>;

} // namespace initializers
