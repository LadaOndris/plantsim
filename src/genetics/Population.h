//
// Created by lada on 1/27/22.
//

#ifndef PLANTSIM_POPULATION_H
#define PLANTSIM_POPULATION_H

#include <memory>
#include <utility>
#include <algorithm>    // std::sort
#include <vector>       // std::vector
#include <random>
#include "genetics/Individual.h"
#include "genetics/IFitness.h"

template<typename TIndividual>
class Population {
public:
    explicit Population(std::unique_ptr<IFitness<TIndividual>> fitness, int size, int chromosomeLength);

    std::vector<TIndividual> &getIndividuals();

    double getMaxFitness() const;

    void evaluate();

    void select();

    void crossover();

    void mutate();


private:
    std::unique_ptr<IFitness<TIndividual>> fitness;
    std::vector<TIndividual> individuals;
    TIndividual maxFitnessIndividual;
    int size;
};

template<typename TIndividual>
Population<TIndividual>::Population(std::unique_ptr<IFitness<TIndividual>> fitness, int size, int chromosomeLength) :
        fitness(std::move(fitness)), individuals(size, TIndividual(chromosomeLength)), size(size) {

}

template<typename TIndividual>
void Population<TIndividual>::evaluate() {
    for (TIndividual &ind : this->individuals) {
        double ind_fitness = this->fitness->compute(ind);
        ind.setFitness(ind_fitness);
        if (ind_fitness > maxFitnessIndividual.getFitness()) {
            maxFitnessIndividual = ind;
        }
    }
}

template<typename TIndividual>
std::vector<TIndividual> &Population<TIndividual>::getIndividuals() {
    return individuals;
}

template<typename TIndividual>
double Population<TIndividual>::getMaxFitness() const {
    return maxFitnessIndividual.getFitness();
}

template<typename TIndividual>
void Population<TIndividual>::select() {
    std::sort(individuals.begin(), individuals.end());
    auto populationSize = individuals.size();
    // Select 10% of population
    int selectionSize = static_cast<int>(populationSize * 0.5);
    individuals.erase(individuals.begin() + selectionSize, individuals.end());
}

template<typename TIndividual>
void Population<TIndividual>::crossover() {
    int poolSize = individuals.size();
    // Define random generation
    std::random_device rd; // Obtain a random number from hardware
    std::mt19937 gen(rd()); // Seed the generator
    std::uniform_int_distribution<int> poolDistribution(0, poolSize - 1); // Define the range

    while (individuals.size() < size) {
        int index1 = poolDistribution(gen);
        int index2 = poolDistribution(gen);
        auto &individual1 = individuals[index1];
        auto &individual2 = individuals[index2];
        auto individualPtr = individual1.crossover(individual2);
        auto testIndividualPtr = dynamic_cast<std::unique_ptr<TIndividual>>(individualPtr);
        //auto individual = individualPtr.release();
        individuals.push_back(*testIndividualPtr.get());
    }
}

template<typename TIndividual>
void Population<TIndividual>::mutate() {
    return;
}


#endif //PLANTSIM_POPULATION_H
