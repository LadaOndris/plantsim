//
// Created by lada on 8/26/21.
//

#ifndef PLANTSIM_SOIL_H
#define PLANTSIM_SOIL_H


#include "Point.h"

class Soil : public Point {
public:
    Soil(int nutrients, int water);
private:
    int nutrients;
    int water;
};


#endif //PLANTSIM_SOIL_H
