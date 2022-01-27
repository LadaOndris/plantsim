//
// Created by lada on 1/27/22.
//

#ifndef PLANTSIM_IFITNESS_H
#define PLANTSIM_IFITNESS_H


template<typename TIndividual>
class IFitness {
public:
    virtual void compute(TIndividual &individual) = 0;
protected:
    virtual ~IFitness() = 0;
};


#endif //PLANTSIM_IFITNESS_H
