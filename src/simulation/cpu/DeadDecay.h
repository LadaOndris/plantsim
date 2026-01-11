#pragma once

#include "simulation/cpu/GridShiftHelper.h"
#include "simulation/State.h"
#include "simulation/Options.h"
#include "simulation/CellState.h"

/**
 * @brief Simulation stage for dead cell decay and nutrient recycling.
 * 
 * Dead cells gradually release their stored water and minerals back into
 * adjacent soil tiles, completing the nutrient cycle. This prevents resources
 * from being permanently locked up in dead matter.
 */
class DeadDecay {
public:
    using MatrixXf = GridShiftHelper::MatrixXf;
    using MatrixXi = GridShiftHelper::MatrixXi;

    explicit DeadDecay(const GridShiftHelper& grid);

    void step(State& state, State& backBuffer, const Options& options);

private:
    const GridShiftHelper& grid;
    
    MatrixXf deadMask;
    MatrixXf soilMask;
    MatrixXf soilNeighborCount;
    MatrixXf releasedWater;
    MatrixXf releasedMineral;
    MatrixXf sharePerSoil;
    MatrixXf waterShare;
    MatrixXf mineralShare;
    MatrixXf waterIncome;
    MatrixXf mineralIncome;
    MatrixXf tempBuffer;
    MatrixXf dirSoilMask;
    MatrixXf waterToSend;
    MatrixXf mineralToSend;
};
