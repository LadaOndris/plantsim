//
// Created by lada on 1/27/22.
//


#ifndef PLANTSIM_SUMFITNESS_H
#define PLANTSIM_SUMFITNESS_H

#include "IFitness.h"
#include "Population.h"

class SumFitness : IFitness {
    void compute_individual(Individual &individual) override;

};


#endif //PLANTSIM_SUMFITNESS_H
