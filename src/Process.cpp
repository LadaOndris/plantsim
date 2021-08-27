//
// Created by lada on 8/27/21.
//

#include "Process.h"
#include <stdexcept>


Process::~Process() = default;


int Process::getGenesCount() const {
    int genesCount =  doGetGenesCount();
    if (genesCount <= 0)
        throw std::range_error("genesCount must be at least 1");
    return genesCount;
}

void Process::invoke(WorldState &worldState, std::shared_ptr<Cell> &cell) {
    doGetGenesCount();
}
