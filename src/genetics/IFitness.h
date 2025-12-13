#pragma once

#include "genetics/Individual.h"

template<typename TGene>
class IFitness {
public:
    virtual double compute(Individual<TGene> &individual) = 0;
    virtual ~IFitness() = default;
};


