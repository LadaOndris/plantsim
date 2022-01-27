//
// Created by lada on 8/26/21.
//

#include "Constraint.h"

Constraint::~Constraint() = default;

void Constraint::update(WorldState &state) const {
    doUpdate(state);
}
