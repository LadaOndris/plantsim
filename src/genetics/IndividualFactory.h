#pragma once

#include <memory>
#include "Individual.h"

template<typename TGene>
class IndividualFactory {
public:
    IndividualFactory() = default;

    virtual std::shared_ptr<Individual<TGene>> create(int chromosomeLength) const = 0;

    virtual ~IndividualFactory() {};
};


