//
// Created by lada on 9/5/21.
//

#ifndef PLANTSIM_EMPTYPROCESS_H
#define PLANTSIM_EMPTYPROCESS_H

#include <iostream>
#include "Process.h"
#include "Cell.h"
#include "WorldState.h"

class EmptyProcess : public Process {
private:
    int doGetGenesCount() const override {
        // The process requires two genes to define its operation.
        return 2;
    }

    void doInvoke(WorldState &worldState, std::shared_ptr<Cell> &cell) override {
        std::cout << "Invoking empty process." << std::endl;
    }
};

#endif //PLANTSIM_EMPTYPROCESS_H
