//
// Created by lada on 1/27/22.
//

#ifndef PLANTSIM_INDIVIDUAL_H
#define PLANTSIM_INDIVIDUAL_H

#include "genetics/Chromosome.h"
#include <limits>


template<typename TGene>
class Individual {
public:
    explicit Individual(int chromosomeLength);

    double getFitness();

    virtual Individual<TGene> crossover(Individual<TGene> &lhs, Individual<TGene> &rhs) = 0;

    virtual Individual<TGene> mutate(Individual<TGene> &individual) = 0;

    Chromosome<TGene> getChromosome();

private:
    Chromosome<TGene> chromosome;
    double fitness;
};

template<typename TGene>
Individual<TGene>::Individual(int chromosomeLength) : chromosome(chromosomeLength) {
    fitness = -std::numeric_limits<double>::infinity();
}


template<typename TGene>
Chromosome<TGene> Individual<TGene>::getChromosome() {
    return chromosome;
}

template <typename TGene>
double Individual<TGene>::getFitness() {
    return fitness;
}

#endif //PLANTSIM_INDIVIDUAL_H
