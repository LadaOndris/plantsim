#pragma once

#include "simulation/cpu/GridShiftHelper.h"
#include "simulation/State.h"
#include "simulation/Options.h"
#include "simulation/CellState.h"
#include <random>
#include <array>

/**
 * @brief Random neighbor reproduction using a three-phase vectorized approach.
 * 
 * Algorithm:
 * 1. **Intention**: Each eligible cell randomly picks one empty neighbor direction
 * 2. **Resolution**: Conflicts resolved
 * 3. **Application**: Winners pay cost, children are created
 */
class RandomNeighborReproduction {
public:
    explicit RandomNeighborReproduction(const GridShiftHelper& grid, 
                                         unsigned int seed = std::random_device{}());

    void step(State& state, State& backBuffer, const Options& options);

private:
    using MatrixXf = GridShiftHelper::MatrixXf;
    using MatrixXi = GridShiftHelper::MatrixXi;
    static constexpr int NUM_DIRECTIONS = GridShiftHelper::NUM_DIRECTIONS;

    const GridShiftHelper& grid;
    std::mt19937 rng;

    // Core masks
    MatrixXf emptyMask;
    MatrixXf eligibleMask;
    MatrixXf emptyNeighborCount;

    // Direction selection (intention phase)
    std::array<MatrixXf, NUM_DIRECTIONS> directionAvailable;
    std::array<MatrixXf, NUM_DIRECTIONS> directionChosen;
    MatrixXf randomValues;
    MatrixXi randomIndex;
    MatrixXi cumulativeCount;  // Integer counter for vectorization
    MatrixXf tempBuffer;
    MatrixXi eligibleInt;      // Pre-computed eligible mask as int

    // Results
    MatrixXf parentCost;
    MatrixXf childMask;

    void initBuffers();
    void intentionPhase(const State& state, const Options& options);
    void resolutionPhase();
    void applicationPhase(State& state, State& backBuffer, const Options& options);
};
