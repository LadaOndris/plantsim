//
// Created by lada on 8/25/21.
//

#include "Environment.h"

void Environment::applyConstraints(WorldState &worldState) {
    for (const auto &constraint: constraints) {
        constraint.update(worldState);
    }
}

Environment::Environment()
        : constraints() {

}
