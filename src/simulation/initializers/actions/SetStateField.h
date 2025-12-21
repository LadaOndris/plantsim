#pragma once

#include "simulation/GridTopology.h"
#include "simulation/State.h"
#include "simulation/CellState.h"

namespace initializers {

/**
 * @brief Generic action policy that sets a State field value using member pointers.
 * 
 * @tparam FieldType The type stored in the vector (float or int)
 * @tparam AmountPolicy A policy type that provides FieldType compute(AxialCoord, GridTopology&)
 */
template<typename FieldType, typename AmountPolicy>
class SetStateField {
public:
    using FieldPtr = std::vector<FieldType> State::*;
    
    SetStateField(AmountPolicy amount, FieldPtr field) 
        : amountPolicy(std::move(amount)), fieldPtr(field) {}

    void apply(AxialCoord coord, const GridTopology& topology, State& state) const {
        OffsetCoord offset = axialToOddr(coord);
        int index = offset.row * topology.width + offset.col;
        (state.*fieldPtr)[index] = static_cast<FieldType>(amountPolicy.compute(coord, topology));
    }
    
private:
    AmountPolicy amountPolicy;
    FieldPtr fieldPtr;
};

template<typename FieldType, typename AmountPolicy>
SetStateField(AmountPolicy, std::vector<FieldType> State::*) 
    -> SetStateField<FieldType, AmountPolicy>;

/**
 * @brief Set plant sugar values for cells.
 * 
 * @tparam AmountPolicy A policy type that provides float compute(AxialCoord, GridTopology&)
 */
template<typename AmountPolicy>
auto SetResource(AmountPolicy amount) {
    return SetStateField<float, AmountPolicy>(std::move(amount), &State::plantSugar);
}

/**
 * @brief Set plant sugar values for cells (alias for SetResource).
 */
template<typename AmountPolicy>
auto SetPlantSugar(AmountPolicy amount) {
    return SetStateField<float, AmountPolicy>(std::move(amount), &State::plantSugar);
}

/**
 * @brief Set plant water values for cells.
 */
template<typename AmountPolicy>
auto SetPlantWater(AmountPolicy amount) {
    return SetStateField<float, AmountPolicy>(std::move(amount), &State::plantWater);
}

/**
 * @brief Set plant mineral values for cells.
 */
template<typename AmountPolicy>
auto SetPlantMineral(AmountPolicy amount) {
    return SetStateField<float, AmountPolicy>(std::move(amount), &State::plantMineral);
}

/**
 * @brief Set soil water values for cells.
 */
template<typename AmountPolicy>
auto SetSoilWater(AmountPolicy amount) {
    return SetStateField<float, AmountPolicy>(std::move(amount), &State::soilWater);
}

/**
 * @brief Set soil mineral values for cells.
 */
template<typename AmountPolicy>
auto SetSoilMineral(AmountPolicy amount) {
    return SetStateField<float, AmountPolicy>(std::move(amount), &State::soilMineral);
}


} // namespace initializers
