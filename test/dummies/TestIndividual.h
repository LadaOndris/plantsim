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
public:
    TestIndividual<TGene>() : Individual<TGene>()  {
    }

    explicit TestIndividual(int chromosomeLength) : Individual<TGene>(chromosomeLength) {
    }

    std::unique_ptr<Individual<TGene>> crossover(Individual<TGene> &other) override {
        return std::make_unique<TestIndividual<TGene>>(0);
    }

    std::unique_ptr<Individual<TGene>> mutate() override {
        return std::make_unique<TestIndividual<TGene>>(0);
    }
};


#endif //PLANTSIM_TESTINDIVIDUAL_H
