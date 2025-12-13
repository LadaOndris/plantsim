#pragma once

#include "genetics/Chromosome.h"
#include <limits>


template<typename TGene>
class Individual {
public:
    Individual();

    explicit Individual(int chromosomeLength);

    double getFitness() const;

    Chromosome<TGene> &getChromosome();

    virtual std::shared_ptr<Individual<TGene>> crossover(Individual<TGene> &other) = 0;

    virtual void mutate() = 0;

    void setFitness(double value);

    template<typename T>
    friend bool operator<(const Individual<T>& lhs, const Individual<T>& rhs);
protected:
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
bool operator<(const Individual<TGene> &lhs, const Individual<TGene> &rhs) {
    return lhs.getFitness() < rhs.getFitness();
}


