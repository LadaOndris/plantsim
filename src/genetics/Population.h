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

    std::vector<TIndividual> getIndividuals() const;

    double getMaxFitness() const;

    void evaluate();

    void select();

    void crossover();

    void mutate();


private:
    std::unique_ptr<IFitness<TIndividual>> fitness;
    std::vector<TIndividual> individuals;
    TIndividual maxFitnessIndividual;
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
std::vector<TIndividual> Population<TIndividual>::getIndividuals() const {
    return individuals;
}

template<typename TIndividual>
double Population<TIndividual>::getMaxFitness() const {
    return maxFitnessIndividual.getFitness();
}

template<typename TIndividual>
void Population<TIndividual>::select() {
    return;
}

template<typename TIndividual>
void Population<TIndividual>::crossover() {
    return;
}

template<typename TIndividual>
void Population<TIndividual>::mutate() {
    return;
}


#endif //PLANTSIM_POPULATION_H
