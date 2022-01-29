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
#include "genetics/IndividualFactory.h"

template<typename TGene>
class Population {
public:
    explicit Population(std::unique_ptr<IFitness<TGene>> fitness, int size,
                        const IndividualFactory<TGene> &factory, int chromosomeLength);

    std::vector<std::shared_ptr<Individual<TGene>>> &getIndividuals();

    double getMaxFitness() const;

    void evaluate();

    void select();

    void crossover();

    void mutate();


private:
    std::unique_ptr<IFitness<TGene>> fitness;
    std::vector<std::shared_ptr<Individual<TGene>>> individuals;
    std::shared_ptr<Individual<TGene>> maxFitnessIndividual;
    int size;
};

template<typename TGene>
Population<TGene>::Population(std::unique_ptr<IFitness<TGene>> fitness, int size,
                              const IndividualFactory<TGene> &factory, int chromosomeLength) :
        fitness(std::move(fitness)), individuals(), size(size) {
    for (int i = 0; i < size; i++) {
        auto individual = factory.create(chromosomeLength);
        individuals.push_back(individual);
    }
}

template<typename TGene>
void Population<TGene>::evaluate() {
    for (std::shared_ptr<Individual<TGene>> &ind : this->individuals) {
        double ind_fitness = this->fitness->compute(*ind);
        ind->setFitness(ind_fitness);
        if (maxFitnessIndividual == nullptr || ind_fitness > maxFitnessIndividual->getFitness()) {
            maxFitnessIndividual = ind;
        }
    }
}

template<typename TGene>
std::vector<std::shared_ptr<Individual<TGene>>> &Population<TGene>::getIndividuals() {
    return individuals;
}

template<typename TGene>
double Population<TGene>::getMaxFitness() const {
    return maxFitnessIndividual->getFitness();
}

template<typename TGene>
void Population<TGene>::select() {
    std::sort(individuals.begin(), individuals.end());
    auto populationSize = individuals.size();
    // Select 10% of population
    int selectionSize = static_cast<int>(populationSize * 0.5);
    individuals.erase(individuals.begin() + selectionSize, individuals.end());
}

template<typename TGene>
void Population<TGene>::crossover() {
    int poolSize = individuals.size();
    // Define random generation
    std::random_device rd; // Obtain a random number from hardware
    std::mt19937 gen(rd()); // Seed the generator
    std::uniform_int_distribution<int> poolDistribution(0, poolSize - 1); // Define the range

    while (individuals.size() < size) {
        int index1 = poolDistribution(gen);
        int index2 = poolDistribution(gen);
        auto individual1 = individuals[index1];
        auto individual2 = individuals[index2];
        auto individualPtr = individual1->crossover(*individual2);
        individuals.push_back(individualPtr);
    }
}

template<typename TGene>
void Population<TGene>::mutate() {
    return;
}


#endif //PLANTSIM_POPULATION_H
