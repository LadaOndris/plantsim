#pragma once

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

