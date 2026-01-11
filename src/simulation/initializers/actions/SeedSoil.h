#pragma once

#include "simulation/GridTopology.h"
#include "simulation/State.h"
#include "simulation/CellState.h"

namespace initializers {

template<typename WaterAmountPolicy, typename MineralAmountPolicy>
class SeedSoil {
public:
    SeedSoil(WaterAmountPolicy waterAmount, MineralAmountPolicy mineralAmount) 
        : waterPolicy(std::move(waterAmount))
        , mineralPolicy(std::move(mineralAmount)) {}

    void apply(AxialCoord coord, const GridTopology& topology, State& state) const {
        int index = topology.toStorageIndex(coord);
        state.cellTypes[index] = static_cast<int>(CellState::Type::Soil);
        state.soilWater[index] = waterPolicy.compute(coord, topology);
        state.soilMineral[index] = mineralPolicy.compute(coord, topology);
    }
    
private:
    WaterAmountPolicy waterPolicy;
    MineralAmountPolicy mineralPolicy;
};

template<typename WaterAmountPolicy, typename MineralAmountPolicy>
SeedSoil(WaterAmountPolicy, MineralAmountPolicy) -> SeedSoil<WaterAmountPolicy, MineralAmountPolicy>;

} // namespace initializers
