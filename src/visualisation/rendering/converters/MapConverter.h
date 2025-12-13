//
// Created by lada on 4/14/24.
//

#ifndef PLANTSIM_MAPCONVERTER_H
#define PLANTSIM_MAPCONVERTER_H

#include "MeshData.h"
#include "simulation/GridTopology.h"

class MapConverter {

public:
    virtual MeshData convert(const GridTopology &topology) const = 0;

};


#endif //PLANTSIM_MAPCONVERTER_H
