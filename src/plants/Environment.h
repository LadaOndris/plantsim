//
// Created by lada on 8/25/21.
//

#ifndef PLANTSIM_ENVIRONMENT_H
#define PLANTSIM_ENVIRONMENT_H

#include <vector>
#include "WorldState.h"
#include "Constraint.h"

class Environment {
public:
    Environment();
    /**
     * Update the world state by applying
     * each of the constraints.
     *
     * @param worldState The current instance of the world.
     */
    void applyConstraints(WorldState& worldState);
private:
    std::vector<Constraint> constraints;
};


#endif //PLANTSIM_ENVIRONMENT_H
