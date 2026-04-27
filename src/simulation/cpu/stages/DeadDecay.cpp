#include "simulation/cpu/stages/DeadDecay.h"

DeadDecay::DeadDecay(const GridShiftHelper& grid)
    : grid(grid)
{
    const int h = grid.height();
    const int w = grid.width();
    
    deadMask.resize(h, w);
    soilMask.resize(h, w);
    soilNeighborCount.resize(h, w);
    releasedWater.resize(h, w);
    releasedMineral.resize(h, w);
    sharePerSoil.resize(h, w);
    waterShare.resize(h, w);
    mineralShare.resize(h, w);
    waterIncome.resize(h, w);
    mineralIncome.resize(h, w);
    tempBuffer.resize(h, w);
    dirSoilMask.resize(h, w);
    waterToSend.resize(h, w);
    mineralToSend.resize(h, w);
}

void DeadDecay::step(State& state, State& backBuffer, const Options& options) {
    if (!options.enableDeadDecay) {
        return;
    }

    const int h = grid.height();
    const int w = grid.width();
    const float dt = options.dt;
    const auto& validity = grid.getValidityMask();

    Eigen::Map<const MatrixXi> cellTypes(state.cellTypes.data(), h, w);
    Eigen::Map<const MatrixXf> deadWater(state.deadWater.data(), h, w);
    Eigen::Map<const MatrixXf> deadMineral(state.deadMineral.data(), h, w);
    Eigen::Map<const MatrixXf> soilWater(state.soilWater.data(), h, w);
    Eigen::Map<const MatrixXf> soilMineral(state.soilMineral.data(), h, w);
    
    Eigen::Map<MatrixXf> nextDeadWater(backBuffer.deadWater.data(), h, w);
    Eigen::Map<MatrixXf> nextDeadMineral(backBuffer.deadMineral.data(), h, w);
    Eigen::Map<MatrixXf> nextSoilWater(backBuffer.soilWater.data(), h, w);
    Eigen::Map<MatrixXf> nextSoilMineral(backBuffer.soilMineral.data(), h, w);
    
    nextDeadWater = deadWater;
    nextDeadMineral = deadMineral;
    nextSoilWater = soilWater;
    nextSoilMineral = soilMineral;

    deadMask = (validity.array() * 
        (cellTypes.array() == static_cast<int>(CellState::Type::Dead)).cast<float>()).matrix();
    soilMask = (validity.array() * 
        (cellTypes.array() == static_cast<int>(CellState::Type::Soil)).cast<float>()).matrix();

    // Count soil neighbors for each dead cell
    soilNeighborCount.setZero();
    for (int d = 0; d < GridShiftHelper::NUM_DIRECTIONS; ++d) {
        grid.shiftMatrix(soilMask, tempBuffer, grid.getOutgoingShift(d));
        soilNeighborCount += tempBuffer;
    }

    // Calculate amount released this tick (fraction of current stores)
    // Clamp to what's available
    releasedWater = (options.deadDecayRate * dt * deadWater.array())
                     .min(deadWater.array()).matrix();
    releasedMineral = (options.deadDecayRate * dt * deadMineral.array())
                       .min(deadMineral.array()).matrix();

    // Only release from dead cells
    releasedWater.array() *= deadMask.array();
    releasedMineral.array() *= deadMask.array();

    // Deduct released amounts from dead pools
    nextDeadWater.array() -= releasedWater.array();
    nextDeadMineral.array() -= releasedMineral.array();

    // Calculate share per soil neighbor
    // share = 1/soilNeighborCount where count > 0, else 0
    sharePerSoil = (soilNeighborCount.array() > 0.0f)
                    .select(1.0f / soilNeighborCount.array(), 0.0f);

    // Calculate what each dead cell gives to each adjacent soil direction
    // Note: waterShare and mineralShare are the TOTAL amount to distribute,
    // divided by neighbor count. Each direction only receives this if it's soil.
    waterShare = releasedWater.array() * sharePerSoil.array();
    mineralShare = releasedMineral.array() * sharePerSoil.array() * options.deadToSoilBias;

    // Distribute to adjacent soil tiles
    waterIncome.setZero();
    mineralIncome.setZero();
    
    for (int d = 0; d < GridShiftHelper::NUM_DIRECTIONS; ++d) {
        // First check which neighbors in this direction are soil
        grid.shiftMatrix(soilMask, dirSoilMask, grid.getOutgoingShift(d));
        
        // Only send resources if the neighbor in this direction is soil
        waterToSend = waterShare.array() * dirSoilMask.array();
        mineralToSend = mineralShare.array() * dirSoilMask.array();
        
        // Shift from dead cell positions to neighbor positions
        grid.shiftMatrix(waterToSend, tempBuffer, grid.getIncomingShift(d));
        waterIncome += tempBuffer;
        
        grid.shiftMatrix(mineralToSend, tempBuffer, grid.getIncomingShift(d));
        mineralIncome += tempBuffer;
    }

    // Add income only to soil cells
    nextSoilWater.array() += waterIncome.array() * soilMask.array();
    nextSoilMineral.array() += mineralIncome.array() * soilMask.array();

    // Clamp to non-negative
    nextDeadWater = nextDeadWater.cwiseMax(0.0f);
    nextDeadMineral = nextDeadMineral.cwiseMax(0.0f);
    
    // Dead cells with no remaining resources become Air
    Eigen::Map<MatrixXi> nextCellTypes(backBuffer.cellTypes.data(), h, w);
    
    auto emptyDeadMask = (deadMask.array() > 0.5f) && 
                         (nextDeadWater.array() <= 1e-6f) && 
                         (nextDeadMineral.array() <= 1e-6f);
    nextCellTypes = emptyDeadMask.select(
        static_cast<int>(CellState::Type::Air), nextCellTypes);

    std::swap(state.cellTypes, backBuffer.cellTypes);
    std::swap(state.deadWater, backBuffer.deadWater);
    std::swap(state.deadMineral, backBuffer.deadMineral);
    std::swap(state.soilWater, backBuffer.soilWater);
    std::swap(state.soilMineral, backBuffer.soilMineral);
}
