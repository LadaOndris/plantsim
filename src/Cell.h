//
// Created by lada on 8/26/21.
//

#ifndef PLANTSIM_CELL_H
#define PLANTSIM_CELL_H


#include "Point.h"
#include "genetics/Chromosome.h"
#include "Process.h"
#include <memory>

class Cell : public Point {
public:
    explicit Cell(int x, int y);
    ~Cell();

};


#endif //PLANTSIM_CELL_H
