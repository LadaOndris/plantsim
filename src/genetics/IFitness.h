//
// Created by lada on 1/27/22.
//

#ifndef PLANTSIM_IFITNESS_H
#define PLANTSIM_IFITNESS_H

#include "genetics/Individual.h"

template<typename TGene>
class IFitness {
public:
    virtual double compute(Individual<TGene> &individual) = 0;
    virtual ~IFitness() = default;
};


#endif //PLANTSIM_IFITNESS_H
