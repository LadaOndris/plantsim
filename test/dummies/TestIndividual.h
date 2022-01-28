//
// Created by lada on 1/27/22.
//

#ifndef PLANTSIM_TESTINDIVIDUAL_H
#define PLANTSIM_TESTINDIVIDUAL_H

#include "genetics/SumFitness.h"
#include "genetics/Population.h"
#include "genetics/Chromosome.h"
#include "genetics/Gene.h"
#include "genetics/Individual.h"

template<typename TGene>
class TestIndividual : public Individual<TGene> {

    using Individual<TGene>::Individual;

    Individual<TGene> crossover(Individual<TGene> &lhs, Individual<TGene> &rhs) override {
        return Individual<TGene>(0);
    }

    Individual<TGene> mutate(Individual<TGene> &individual) override {
        return Individual<TGene>(0);
    }
};


#endif //PLANTSIM_TESTINDIVIDUAL_H
