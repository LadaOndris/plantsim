#pragma once

#include "simulation/GridTopology.h"
#include <Eigen/Dense>

namespace initializers {

/**
 * @brief Action policy that sets values in an Eigen matrix.
 * 
 * @details This action is designed for use with the Eigen matrix overload
 * of PolicyApplication::apply(). It computes a value using an AmountPolicy
 * and stores it in the matrix at the appropriate position.
 * 
 * @tparam AmountPolicy A policy type that provides float compute(AxialCoord, GridTopology&)
 */
template<typename AmountPolicy>
class SetValue {
public:
    using MatrixXf = GridShiftHelper::MatrixXf;

    explicit SetValue(AmountPolicy amount) : amountPolicy(std::move(amount)) {}

    /**
     * @brief Apply the value to a matrix at the given coordinates.
     * 
     * @param coord The axial coordinate in the grid
     * @param topology The grid topology
     * @param matrix The Eigen matrix to modify (storage layout: row-major)
     */
    void apply(AxialCoord coord, const GridTopology& topology, MatrixXf& matrix) const {
        OffsetCoord offset = axialToOddr(coord);
        matrix(offset.row, offset.col) = amountPolicy.compute(coord, topology);
    }

private:
    AmountPolicy amountPolicy;
};

template<typename AmountPolicy>
SetValue(AmountPolicy) -> SetValue<AmountPolicy>;

} // namespace initializers
