#pragma once

#include "MapConverter.h"

class AxialRectangularMapToMeshConverter : public MapConverter {

public:
    MeshData convert(const GridTopology &topology) const override;

};


