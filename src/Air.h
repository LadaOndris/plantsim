//
// Created by lada on 8/26/21.
//

#ifndef PLANTSIM_AIR_H
#define PLANTSIM_AIR_H


#include "Point.h"

class Air : public Point {
public:
    explicit Air(int x, int y, int oxygen = 0);
private:
    int oxygen;
};


#endif //PLANTSIM_AIR_H
