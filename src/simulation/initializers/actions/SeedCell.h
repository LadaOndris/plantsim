#pragma once

#include "simulation/GridTopology.h"
#include "simulation/State.h"
#include "simulation/CellState.h"

namespace initializers {

template<typename HealthAmountPolicy>
class SeedCell {
public:
    explicit SeedCell(HealthAmountPolicy healthAmount) 
        : amountPolicy(std::move(healthAmount)) {}

    void apply(AxialCoord coord, const GridTopology& topology, State& state) const {
        int index = topology.toStorageIndex(coord);
        state.cellTypes[index] = static_cast<int>(CellState::Type::Cell);
        state.plantHealth[index] = amountPolicy.compute(coord, topology);
    }
    
private:
    HealthAmountPolicy amountPolicy;
};

template<typename AmountPolicy>
SeedCell(AmountPolicy) -> SeedCell<AmountPolicy>;

} // namespace initializers
