#pragma once

#include "WorldState.h"

class Constraint {

public:
    ~Constraint();
    void update(WorldState &state) const;
private:
    virtual void doUpdate(WorldState &state) const = 0;
};


