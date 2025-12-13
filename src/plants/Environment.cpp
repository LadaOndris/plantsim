
#include "Environment.h"

void Environment::applyConstraints(WorldState &worldState) {
    for (const auto &constraint: constraints) {
        constraint.update(worldState);
    }
}

Environment::Environment()
        : constraints() {

}
