#include "simulation/cpu/RandomNeighborReproduction.h"
#include <algorithm>
#include <cmath>

RandomNeighborReproduction::RandomNeighborReproduction(const GridShiftHelper& grid, 
                                                         unsigned int seed)
    : grid(grid)
    , rng(seed)
{
    initBuffers();
}

void RandomNeighborReproduction::initBuffers() {
    const int h = grid.height();
    const int w = grid.width();

    emptyMask.resize(h, w);
    eligibleMask.resize(h, w);
    emptyNeighborCount.resize(h, w);
    randomValues.resize(h, w);
    randomIndex.resize(h, w);
    cumulativeCount.resize(h, w);
    tempBuffer.resize(h, w);
    eligibleInt.resize(h, w);
    
    for (int d = 0; d < NUM_DIRECTIONS; ++d) {
        directionAvailable[d].resize(h, w);
        directionChosen[d].resize(h, w);
    }
    
    parentCost.resize(h, w);
    childMask.resize(h, w);
}

void RandomNeighborReproduction::step(State& state, State& backBuffer, const Options& options) {
    if (!options.enableCellMultiplication) {
        return;
    }

    intentionPhase(state);
    resolutionPhase();
    applicationPhase(state, backBuffer);
}

void RandomNeighborReproduction::intentionPhase(const State& state) {
    const int h = grid.height();
    const int w = grid.width();
    const auto& validity = grid.getValidityMask();
    
    Eigen::Map<const MatrixXf> resources(state.resources.data(), h, w);
    Eigen::Map<const MatrixXi> cellTypes(state.cellTypes.data(), h, w);

    auto valid = validity.array();
    auto isAir  = (cellTypes.array() == static_cast<int>(CellState::Type::Air)).cast<float>();
    auto isCell = (cellTypes.array() == static_cast<int>(CellState::Type::Cell)).cast<float>();
    auto hasResources = (resources.array() >= config.reproductionThreshold).cast<float>();
    auto hasEmptyNeighbor = (emptyNeighborCount.array() > 0).cast<float>();
    
    emptyMask = (valid * isAir).matrix();
    eligibleMask = (valid * isCell * hasResources * hasEmptyNeighbor).matrix();

    // For each direction, check if neighbor in that direction is empty
    emptyNeighborCount.setZero();
    for (int d = 0; d < NUM_DIRECTIONS; ++d) {
        grid.shiftMatrix(emptyMask, directionAvailable[d], grid.getOutgoingShift(d));
        emptyNeighborCount += directionAvailable[d];
    }

    std::uniform_real_distribution<float> dist(0.0f, std::nextafter(1.0f, 0.0f));
    std::generate(randomIndex.data(), randomIndex.data() + grid.size(),
                  [&]() { return dist(rng); });
    randomIndex = (randomValues.array() * emptyNeighborCount.array()).floor().cast<int>();

    eligibleInt = (eligibleMask.array() > 0.5f).cast<int>();
    cumulativeCount.setZero();

    for (int d = 0; d < NUM_DIRECTIONS; ++d) {
        using ArrayXXiRowMajor = Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        ArrayXXiRowMajor dirAvailInt = (directionAvailable[d].array() > 0.5f).cast<int>();

        auto chosen =
            eligibleInt.array()
            * dirAvailInt
            * (cumulativeCount.array() == randomIndex.array()).cast<int>();

        directionChosen[d].array() = chosen.cast<float>();

        // Integer increment for cumulative count
        cumulativeCount.array() += dirAvailInt;
    }
}

void RandomNeighborReproduction::resolutionPhase() {
    const int h = grid.height();
    const int w = grid.width();
    
    childMask.setZero();
    parentCost.setZero();
    
    for (int d = 0; d < NUM_DIRECTIONS; ++d) {
        // Shift parent intentions to target cells
        grid.shiftMatrix(directionChosen[d], tempBuffer, grid.getIncomingShift(d));
        
        // Only consider if target is empty AND not already claimed
        tempBuffer.array() *= emptyMask.array() * (1.0f - childMask.array());
        
        // These cells are now claimed
        childMask.array() += tempBuffer.array();
        
        // Shift back to identify winning parents
        grid.accumulateShifted(parentCost, tempBuffer, grid.getOutgoingShift(d));
    }
}

void RandomNeighborReproduction::applicationPhase(State& state, State& backBuffer) {
    const int h = grid.height();
    const int w = grid.width();
    
    Eigen::Map<const MatrixXf> resources(state.resources.data(), h, w);
    Eigen::Map<const MatrixXi> cellTypes(state.cellTypes.data(), h, w);
    Eigen::Map<MatrixXf> nextResources(backBuffer.resources.data(), h, w);
    Eigen::Map<MatrixXi> nextCellTypes(backBuffer.cellTypes.data(), h, w);

    nextResources = resources;
    nextCellTypes = cellTypes;

    // Deduct cost from winning parents
    nextResources.array() -= parentCost.array() * config.reproductionCost;

    // Create children
    nextCellTypes = (childMask.array() > 0.5f).select(
        static_cast<int>(CellState::Type::Cell), nextCellTypes);
    nextResources = (childMask.array() > 0.5f).select(
        config.childInitialResources, nextResources);

    std::swap(state.resources, backBuffer.resources);
    std::swap(state.cellTypes, backBuffer.cellTypes);
}
