#pragma once

#include "MapConverter.h"

class OffsetRectangularMapToMeshConverter : public MapConverter {

public:
    MeshData convert(const GridTopology &topology) const override;

};


