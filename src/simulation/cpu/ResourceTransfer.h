#pragma once

#include "simulation/cpu/GridShiftHelper.h"
#include "simulation/State.h"
#include "simulation/Options.h"
#include "simulation/CellState.h"

/**
 * @brief Resource transfer between cells using matrix operations.
 * 
 * Each cell with resources distributes them equally to neighboring cells
 * that can receive (valid Cell type). The transfer is symmetric and 
 * conserves total resources.
 */
class ResourceTransfer {
public:
    explicit ResourceTransfer(const GridShiftHelper& grid);

    void step(State& state, State& backBuffer, const Options& options);

private:
    using MatrixXf = GridShiftHelper::MatrixXf;
    using MatrixXi = GridShiftHelper::MatrixXi;

    const GridShiftHelper& grid;

    // Pre-allocated buffers
    MatrixXf receiverMask;
    MatrixXf neighborsCanReceiveCount;
    MatrixXf flowPerNeighbor;
    MatrixXf totalIncoming;
};
