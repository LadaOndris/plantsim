//
// Created by lada on 8/26/21.
//

#ifndef PLANTSIM_CELL_H
#define PLANTSIM_CELL_H


#include "Point.h"
#include "genetics/Chromosome.h"
#include <memory>

class Cell : public Point {
private:
    std::shared_ptr<Chromosome<int>> chromosome;
};


#endif //PLANTSIM_CELL_H
