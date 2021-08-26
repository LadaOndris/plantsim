//
// Created by lada on 8/26/21.
//

#ifndef PLANTSIM_CONSTRAINT_H
#define PLANTSIM_CONSTRAINT_H


#include "WorldState.h"

class Constraint {

public:
    ~Constraint();
    void update(WorldState &state) const;
private:
    virtual void doUpdate(WorldState &state) const = 0;
};


#endif //PLANTSIM_CONSTRAINT_H
