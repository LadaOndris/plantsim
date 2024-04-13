//
// Created by lada on 8/27/21.
//

#include "Process.h"
#include <stdexcept>


Process::~Process() = default;

void Process::invoke(Entity &entity, std::shared_ptr<Cell> &cell) {
    doInvoke(entity, cell);
}
