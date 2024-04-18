//
// Created by lada on 8/26/21.
//

#include "Cell.h"
#include <iostream>


Cell::Cell(int x, int y) : Point(x, y), resources(0) {
    std::cout << "Constructing a cell." << std::endl;
}

Cell::~Cell() {
    std::cout << "Destructing a cell." << std::endl;
}
