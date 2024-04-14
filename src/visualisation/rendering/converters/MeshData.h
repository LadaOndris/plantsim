//
// Created by lada on 4/14/24.
//

#ifndef PLANTSIM_MESHDATA_H
#define PLANTSIM_MESHDATA_H

#include <vector>
#include "visualisation/rendering/GLVertex.h"


struct MeshData {
    std::vector<GLVertex> vertices;
    std::vector<unsigned int> indices;
};



#endif //PLANTSIM_MESHDATA_H
