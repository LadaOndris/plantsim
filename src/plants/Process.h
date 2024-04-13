//
// Created by lada on 8/27/21.
//

#ifndef PLANTSIM_PROCESS_H
#define PLANTSIM_PROCESS_H

#include "Entity.h"
#include "Cell.h"

class WorldState;

class Process {
public:

    virtual ~Process();

    void invoke(Entity &entity, std::shared_ptr<Cell> &cell);

private:
    virtual void doInvoke(Entity &entity, std::shared_ptr<Cell> &cell) = 0;

};


#endif //PLANTSIM_PROCESS_H
