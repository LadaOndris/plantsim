#pragma once

#include "Point.h"

class Soil : public Point {
public:
    Soil(int x, int y, int nutrients, int water);
private:
    int nutrients;
    int water;
};


