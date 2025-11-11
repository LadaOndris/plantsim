

#pragma once

#include "plants/Point.h"

/**
 * Represents the state of a single cell in the simulation.
 * Contains application-specific data associated with each cell in the map.
 * Fields are ordered to minimize padding.
 */
struct CellState {
    enum Type : uint8_t {
        Air = 0,
        Cell = 1
    };

    int resources;
    Type type;
    uint8_t valid;

    CellState() : resources(0), type(Type::Air), valid(0) {}

    explicit CellState(Type type, int resources = 0, uint8_t valid = 1) 
        : resources(resources), type(type), valid(valid) {}
};

