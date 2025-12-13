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
 * 2. **Resolution**: Conflicts resolved by direction priority (lowest index wins)
 * 3. **Application**: Winners pay cost, children are created
 */
class RandomNeighborReproduction {
public:
    struct Config {
        float reproductionThreshold = 10.0f;
        float reproductionCost = 5.0f;
        float childInitialResources = 3.0f;
    };

    explicit RandomNeighborReproduction(const GridShiftHelper& grid, 
                                         unsigned int seed = std::random_device{}());

    void step(State& state, State& backBuffer, const Options& options);

    void setConfig(const Config& cfg) { config = cfg; }
    [[nodiscard]] const Config& getConfig() const { return config; }

private:
    using MatrixXf = GridShiftHelper::MatrixXf;
    using MatrixXi = GridShiftHelper::MatrixXi;
    static constexpr int NUM_DIRECTIONS = GridShiftHelper::NUM_DIRECTIONS;

    const GridShiftHelper& grid;
    Config config;
    std::mt19937 rng;

    // Core masks
    MatrixXf emptyMask;
    MatrixXf eligibleMask;
    MatrixXf emptyNeighborCount;

    // Direction selection (intention phase)
    std::array<MatrixXf, NUM_DIRECTIONS> directionAvailable;
    std::array<MatrixXf, NUM_DIRECTIONS> directionChosen;
    MatrixXf randomValues;
    MatrixXf cumulativeProb;
    MatrixXf tempBuffer;

    // Results
    MatrixXf parentCost;
    MatrixXf childMask;

    void initBuffers();
    void intentionPhase(const State& state);
    void resolutionPhase();
    void applicationPhase(State& state, State& backBuffer);
};
