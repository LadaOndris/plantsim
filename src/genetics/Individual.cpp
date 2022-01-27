//
// Created by lada on 1/27/22.
//

#include "Individual.h"

double Individual::getFitness() {
    return this.maxFitness;
}

Individual::Individual(int chromosomeLength) : chromosome(chromosomeLength) {

}
