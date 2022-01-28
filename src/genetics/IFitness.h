//
// Created by lada on 1/27/22.
//

#ifndef PLANTSIM_IFITNESS_H
#define PLANTSIM_IFITNESS_H


template<typename TIndividual>
class IFitness {
public:
    virtual double compute(TIndividual &individual) = 0;
    virtual ~IFitness() = default;
};


#endif //PLANTSIM_IFITNESS_H
