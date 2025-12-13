#pragma once

#include "IFitness.h"
#include "Population.h"
#include "Individual.h"

template<typename TGene>
class SumFitness : public IFitness<TGene> {
    double compute(Individual<TGene> &individual) override;

};

template<typename TGene>
double SumFitness<TGene>::compute(Individual<TGene> &individual) {
    double fitnessSum = 0;
    for (auto &gene : individual.getChromosome().getGenes()) {
        fitnessSum += gene.getValue();
    }
    return fitnessSum;
}


