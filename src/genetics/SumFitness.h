//
// Created by lada on 1/27/22.
//


#ifndef PLANTSIM_SUMFITNESS_H
#define PLANTSIM_SUMFITNESS_H

#include "IFitness.h"
#include "Population.h"

template<typename TIndividual>
class SumFitness : public IFitness<TIndividual> {
    double compute(TIndividual &individual) override;

};

template<typename TIndividual>
double SumFitness<TIndividual>::compute(TIndividual &individual) {
    double fitnessSum = 0;
    for (auto &gene : individual.getChromosome().getGenes()) {
        fitnessSum += gene.getValue();
    }
    return fitnessSum;
}


#endif //PLANTSIM_SUMFITNESS_H
