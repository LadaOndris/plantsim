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
    Individual<TGene>();

    explicit Individual(int chromosomeLength);

    double getFitness() const;

    Chromosome<TGene> &getChromosome();

    virtual std::unique_ptr<Individual<TGene>> crossover(Individual<TGene> &lhs, Individual<TGene> &rhs) = 0;

    virtual std::unique_ptr<Individual<TGene>> mutate(Individual<TGene> &individual) = 0;

    void setFitness(double value);

    friend bool operator>(const Individual<TGene>& lhs, const Individual<TGene>& rhs);
private:
    Chromosome<TGene> chromosome;
    double fitness;
};

template<typename TGene>
Individual<TGene>::Individual() : fitness(0), chromosome(0) {
}

template<typename TGene>
Individual<TGene>::Individual(int chromosomeLength) : chromosome(chromosomeLength) {
    fitness = -std::numeric_limits<double>::infinity();
}

template<typename TGene>
Chromosome<TGene> &Individual<TGene>::getChromosome() {
    return chromosome;
}

template<typename TGene>
double Individual<TGene>::getFitness() const {
    return fitness;
}

template<typename TGene>
void Individual<TGene>::setFitness(double value) {
    fitness = value;
}

template<typename TGene>
bool operator>(const Individual<TGene> &lhs, const Individual<TGene> &rhs) {
    return lhs.getFitness() > rhs.getFitness();
}


#endif //PLANTSIM_INDIVIDUAL_H
