//
// Created by lada on 8/26/21.
//

#ifndef PLANTSIM_WORLDSTATE_H
#define PLANTSIM_WORLDSTATE_H

#include <memory>
#include <array>
#include <boost/scoped_array.hpp>
#include "Point.h"

class WorldState {
public:
    WorldState(int width, int height);
private:
    typedef boost::scoped_array<Point> scoped_array;
    scoped_array map;
};


#endif //PLANTSIM_WORLDSTATE_H
