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
    cumulativeProb.resize(h, w);
    tempBuffer.resize(h, w);
    
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

    // Identify empty cells
    emptyMask = (validity.array() * 
                 (cellTypes.array() == static_cast<int>(CellState::Type::Air)).cast<float>()).matrix();

    // For each direction, check if neighbor in that direction is empty
    emptyNeighborCount.setZero();
    for (int d = 0; d < NUM_DIRECTIONS; ++d) {
        grid.shiftMatrix(emptyMask, directionAvailable[d], grid.getOutgoingShift(d));
        emptyNeighborCount += directionAvailable[d];
    }

    // Eligible = valid Cell with enough resources and at least one empty neighbor
    auto isCell = (cellTypes.array() == static_cast<int>(CellState::Type::Cell)).cast<float>();
    auto hasResources = (resources.array() >= config.reproductionThreshold).cast<float>();
    auto hasEmptyNeighbor = (emptyNeighborCount.array() > 0).cast<float>();
    
    eligibleMask = (validity.array() * isCell * hasResources * hasEmptyNeighbor).matrix();

    // Generate random values in [0, 1)
    std::uniform_real_distribution<float> dist(0.0f, std::nextafter(1.0f, 0.0f));
    std::generate(randomValues.data(), randomValues.data() + grid.size(),
                  [&]() { return dist(rng); });

    // Select one direction per eligible cell using cumulative probability
    auto invCount = (emptyNeighborCount.array() > 0).select(1.0f / emptyNeighborCount.array(), 0.0f);
    
    cumulativeProb.setZero();
    for (int d = 0; d < NUM_DIRECTIONS; ++d) {
        tempBuffer = cumulativeProb;  // prevCumulative
        cumulativeProb.array() += directionAvailable[d].array() * invCount;
        
        // Chosen if: eligible AND available AND prev <= random < curr
        directionChosen[d] = (eligibleMask.array() > 0.5f &&
                              directionAvailable[d].array() > 0.5f &&
                              randomValues.array() >= tempBuffer.array() &&
                              randomValues.array() < cumulativeProb.array()).cast<float>();
    }
}

void RandomNeighborReproduction::resolutionPhase() {
    const int h = grid.height();
    const int w = grid.width();
    
    childMask.setZero();
    parentCost.setZero();
    
    MatrixXf winningParents(h, w);
    
    for (int d = 0; d < NUM_DIRECTIONS; ++d) {
        // Shift parent intentions to target cells
        grid.shiftMatrix(directionChosen[d], tempBuffer, grid.getIncomingShift(d));
        
        // Only consider if target is empty AND not already claimed
        tempBuffer.array() *= emptyMask.array() * (1.0f - childMask.array());
        
        // These cells are now claimed
        childMask.array() += tempBuffer.array();
        
        // Shift back to identify winning parents
        grid.shiftMatrix(tempBuffer, winningParents, grid.getOutgoingShift(d));
        
        // Only parents who chose this direction pay
        parentCost.array() += winningParents.array() * directionChosen[d].array();
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
