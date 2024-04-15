//
// Created by lada on 8/26/21.
//

#ifndef PLANTSIM_POINT_H
#define PLANTSIM_POINT_H

#include <utility>

class Point {
public:
    explicit Point(int x, int y);
    virtual ~Point();
    int getX() const;
    int getY() const;
    std::pair<int, int> getCoords() const;
private:
    std::pair<int, int> coords;
};


#endif //PLANTSIM_POINT_H
