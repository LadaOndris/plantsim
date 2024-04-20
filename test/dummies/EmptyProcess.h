//
// Created by lada on 9/5/21.
//

#ifndef PLANTSIM_EMPTYPROCESS_H
#define PLANTSIM_EMPTYPROCESS_H

#include <iostream>
#include "plants/Process.h"
#include "plants/Cell.h"
#include "plants/Entity.h"

class EmptyProcess : public Process {
private:

    virtual void doInvoke(Entity &entity, Point &cell) {
        auto chromosome = entity.getChromosome();
        
        std::cout << "Invoking empty process." << std::endl;
    }
};

#endif //PLANTSIM_EMPTYPROCESS_H
