#pragma once

#include "simulation/GridTopology.h"
#include <functional>

namespace initializers {

/**
 * @brief Amount policy that uses a custom function to compute values.
 * 
 * @details The function receives the axial coordinate and topology,
 * allowing arbitrary value computation.
 */
class FunctionAmount {
public:
    using ComputeFunction = std::function<float(AxialCoord, const GridTopology&)>;

    ComputeFunction function;

    explicit FunctionAmount(ComputeFunction func) : function(std::move(func)) {}

    [[nodiscard]] float compute(AxialCoord coord, const GridTopology& topology) const {
        return function(coord, topology);
    }
};

} // namespace initializers
