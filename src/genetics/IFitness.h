//
// Created by lada on 1/27/22.
//

#ifndef PLANTSIM_IFITNESS_H
#define PLANTSIM_IFITNESS_H


class IFitness {
public:
    void compute(Individual &individual) = 0;
protected:
    virtual ~IFitness() = 0;
};


#endif //PLANTSIM_IFITNESS_H
