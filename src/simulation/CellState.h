

#pragma once

#include "plants/Point.h"

/**
 * Represents the state of a single cell in the simulation.
 * Contains application-specific data associated with each cell in the map.
 */
struct CellState {
    enum Type {
        Air = 0,
        Cell = 1
    };

    Type type;
    int resources;

    CellState() : type(Type::Air), resources(0) {}

    explicit CellState(Type type, int resources = 0) 
        : type(type), resources(resources) {}
};

