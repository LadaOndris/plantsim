//
// Created by lada on 1/29/22.
//

#ifndef PLANTSIM_INDIVIDUALFACTORY_H
#define PLANTSIM_INDIVIDUALFACTORY_H

#include <memory>
#include "Individual.h"

template<typename TGene>
class IndividualFactory {
public:
    IndividualFactory() = default;

    virtual std::shared_ptr<Individual<TGene>> create(int chromosomeLength) const = 0;

    virtual ~IndividualFactory() {};
};


#endif //PLANTSIM_INDIVIDUALFACTORY_H
