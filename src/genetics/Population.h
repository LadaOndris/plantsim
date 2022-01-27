//
// Created by lada on 1/27/22.
//

#ifndef PLANTSIM_POPULATION_H
#define PLANTSIM_POPULATION_H

#include <memory>
#include <utility>
#include "genetics/Individual.h"
#include "genetics/IFitness.h"

template<typename TIndividual>
class Population {
public:
    explicit Population(std::unique_ptr<IFitness<TIndividual>> fitness, int size, int chromosomeLength);
    void evaluate();
    std::vector<TIndividual> getIndividuals();
    Population<TIndividual> select();
    Population<TIndividual> crossover();
    Population<TIndividual> mutate();
    double getMaxFitness();

private:
    std::unique_ptr<IFitness<TIndividual>> fitness;
    std::vector<TIndividual> individuals;
    TIndividual maxFitnessIndividual = nullptr;
};

template<typename TIndividual>
Population<TIndividual>::Population(std::unique_ptr<IFitness<TIndividual>> fitness, int size, int chromosomeLength) :
fitness(std::move(fitness)), individuals(size, TIndividual(chromosomeLength)) {

}

template<typename TIndividual>
void Population<TIndividual>::evaluate() {
    for (TIndividual &ind : this->individuals) {
        double ind_fitness = this->fitness->compute(ind);
        ind.setFitness(ind_fitness);
        maxFitnessIndividual = ind;
    }
}

template<typename TIndividual>
std::vector<TIndividual> Population<TIndividual>::getIndividuals() {
    return individuals;
}

template<typename TIndividual>
double Population<TIndividual>::getMaxFitness() {
    if (maxFitnessIndividual != 0)
        return maxFitnessIndividual.getFitness();
    return 0;
}

template<typename TIndividual>
Population<TIndividual> Population<TIndividual>::select() {
    return this;
}

template<typename TIndividual>
Population<TIndividual> Population<TIndividual>::crossover() {
    return this;
}

template<typename TIndividual>
Population<TIndividual> Population<TIndividual>::mutate() {
    return this;
}


#endif //PLANTSIM_POPULATION_H
