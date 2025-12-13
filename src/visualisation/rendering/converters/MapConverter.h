#pragma once

#include "MeshData.h"
#include "simulation/GridTopology.h"

class MapConverter {

public:
    virtual MeshData convert(const GridTopology &topology) const = 0;

};


