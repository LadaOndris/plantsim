//
// Created by lada on 4/14/24.
//

#ifndef PLANTSIM_MAPCONVERTER_H
#define PLANTSIM_MAPCONVERTER_H

#include "plants/Map.h"
#include "MeshData.h"

class MapConverter {

public:
    virtual MeshData convert(Map &map) const = 0;

};


#endif //PLANTSIM_MAPCONVERTER_H
