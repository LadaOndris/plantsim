//
// Created by lada on 1/27/22.
//

#ifndef PLANTSIM_INDIVIDUAL_H
#define PLANTSIM_INDIVIDUAL_H

#include "genetics/Chromosome.h"

class Individual {
public:
    Individual(int chromosomeLength);
    double getFitness();
    virtual Individual crossover(Individual &lhs, Individual &rhs) = 0;
    virtual Individual mutate(Individual &individual) = 0;
private:
    Chromosome<int> chromosome;
    double fitness = -std::numeric_limits<double>::infinity();
};


#endif //PLANTSIM_INDIVIDUAL_H
