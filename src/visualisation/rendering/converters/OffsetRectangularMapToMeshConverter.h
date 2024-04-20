//
// Created by lada on 4/14/24.
//

#ifndef PLANTSIM_OFFSETRECTANGULARMAPTOMESHCONVERTER_H
#define PLANTSIM_OFFSETRECTANGULARMAPTOMESHCONVERTER_H

#include "MapConverter.h"

class OffsetRectangularMapToMeshConverter : public MapConverter {

public:
    MeshData convert(Map &map) const override;

};


#endif //PLANTSIM_OFFSETRECTANGULARMAPTOMESHCONVERTER_H
