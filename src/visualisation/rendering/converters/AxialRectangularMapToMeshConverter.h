//
// Created by lada on 4/14/24.
//

#ifndef PLANTSIM_AXIALRECTANGULARMAPTOMESHCONVERTER_H
#define PLANTSIM_AXIALRECTANGULARMAPTOMESHCONVERTER_H

#include "MapConverter.h"
#include "simulation/CellState.h"

class AxialRectangularMapToMeshConverter : public MapConverter {

public:
    MeshData convert(AxialRectangularMap<CellState> &map) const override;

};


#endif //PLANTSIM_AXIALRECTANGULARMAPTOMESHCONVERTER_H
