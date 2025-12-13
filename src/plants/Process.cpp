
#include "Process.h"
#include <stdexcept>


Process::~Process() = default;

void Process::invoke(Entity &entity, Point &cell) {
    doInvoke(entity, cell);
}
