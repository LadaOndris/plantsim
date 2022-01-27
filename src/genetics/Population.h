//
// Created by lada on 1/27/22.
//

#ifndef PLANTSIM_POPULATION_H
#define PLANTSIM_POPULATION_H

#include "genetics/Individual.h"
#include "genetics/IFitness.h"

class Population {
public:
    explicit Population(std::shared_ptr<IFitness> fitness);
    void evaluate();
    std::vector<Individual> getIndividuals();
    virtual Population select() = 0;
    virtual Population crossover() = 0;
    virtual Population mutate() = 0;
    double getMaxFitness();

private:
    std::shared_ptr<IFitness> fitness;
    std::vector<Individual> individuals;
    Individual maxFitnessIndividual = nullptr;
};


#endif //PLANTSIM_POPULATION_H
