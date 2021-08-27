//
// Created by lada on 8/27/21.
//

#ifndef PLANTSIM_PROCESS_H
#define PLANTSIM_PROCESS_H

#include "WorldState.h"
#include "Cell.h"

class Process {
public:

    virtual ~Process();

    void invoke(WorldState &worldState, std::shared_ptr<Cell> &cell);

    int getGenesCount() const;

private:
    virtual void doInvoke(WorldState &worldState, std::shared_ptr<Cell> &cell) = 0;

    virtual int doGetGenesCount() const = 0;
};


#endif //PLANTSIM_PROCESS_H
