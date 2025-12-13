#pragma once

#include "Entity.h"
#include "Cell.h"

class WorldState;

class Process {
public:

    virtual ~Process();

    void invoke(Entity &entity, Point &cell);

private:
    virtual void doInvoke(Entity &entity, Point &cell) = 0;

};


