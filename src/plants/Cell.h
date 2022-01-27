//
// Created by lada on 8/26/21.
//

#ifndef PLANTSIM_CELL_H
#define PLANTSIM_CELL_H


#include <memory>
#include "genetics/Chromosome.h"
#include "Point.h"

class Cell : public Point {
public:
    explicit Cell(int x, int y);
    ~Cell();

};


#endif //PLANTSIM_CELL_H
