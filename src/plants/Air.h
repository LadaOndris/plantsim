#pragma once

#include "Point.h"

class Air : public Point {
public:
    explicit Air(int x, int y, int oxygen = 0);
private:
    int oxygen;
};


