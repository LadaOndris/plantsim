//
// Created by lada on 1/27/22.
//


#ifndef PLANTSIM_SUMFITNESS_H
#define PLANTSIM_SUMFITNESS_H

#include "IFitness.h"
#include "Population.h"

template<typename TIndividual>
class SumFitness : IFitness<TIndividual> {
    void compute(TIndividual &individual) override;

};

template<typename TIndividual>
void SumFitness<TIndividual>::compute(TIndividual &individual) {
    individual.setFitness(0);
}


#endif //PLANTSIM_SUMFITNESS_H
