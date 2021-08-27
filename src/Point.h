//
// Created by lada on 8/26/21.
//

#ifndef PLANTSIM_POINT_H
#define PLANTSIM_POINT_H


class Point {
public:
    explicit Point(int x, int y);
    virtual ~Point();
private:
    int x;
    int y;
};


#endif //PLANTSIM_POINT_H
