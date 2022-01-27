//
// Created by lada on 1/27/22.
//

#include "genetics/Population.h"
#include "genetics/Individual.h"
#include "genetics/IFitness.h"

Population::Population(std::shared_ptr <IFitness> fitness, int size, int chromosomeLength) :
        fitness(fitness), individuals(size, Individual(chromosomeLength)) {

}

void Population::evaluate() {
    for (Individual &ind : this->individuals) {
        double ind_fitness = this->fitness.compute(ind);
        ind.setFitness(ind_fitness);
        maxFitnessIndividual = ind;
    }
}

double Population::getMaxFitness() {
    if (maxFitnessIndividual != 0)
        return maxFitnessIndividual.getFitness();
    return 0;
}

std::vector<Individual> Population::getIndividuals() {
    return individuals;
}
