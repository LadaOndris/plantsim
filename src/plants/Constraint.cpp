
#include "Constraint.h"

Constraint::~Constraint() = default;

void Constraint::update(WorldState &state) const {
    doUpdate(state);
}
