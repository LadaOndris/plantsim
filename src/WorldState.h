//
// Created by lada on 8/26/21.
//

#ifndef PLANTSIM_WORLDSTATE_H
#define PLANTSIM_WORLDSTATE_H

#include <memory>
#include <array>
#include <boost/scoped_array.hpp>
#include "Point.h"
#include "Entity.h"

class WorldState {
public:
    WorldState(int width, int height);
private:
    boost::scoped_array<Point> map;
    /**
     * An entity contains the processes that each cell
     * can do and a chromosome all cells share.
     */
    std::unique_ptr<Entity> entity;
    int width;
    int height;
};


#endif //PLANTSIM_WORLDSTATE_H
