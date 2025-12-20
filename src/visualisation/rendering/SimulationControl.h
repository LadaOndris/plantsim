#pragma once

#include "simulation/Options.h"

/**
 * @brief Control state for simulation playback and configuration.
 */
struct SimulationControl {
    /// Whether the simulation is currently paused
    bool isPaused = false;
    
    /// Flag to request simulation reset (consumed after use)
    bool shouldReset = false;
    
    /// Number of simulation steps per render frame
    int stepsPerFrame = 50;
    
    /// Pending simulation options (applied on reset)
    Options pendingOptions{};
    
    /// Currently active options (for comparison to show pending changes)
    Options activeOptions{};
    
    /**
     * @brief Check if there are pending option changes.
     * @return true if pendingOptions differs from activeOptions
     */
    bool hasPendingChanges() const {
        return pendingOptions.enableResourceTransfer != activeOptions.enableResourceTransfer
            || pendingOptions.enableCellMultiplication != activeOptions.enableCellMultiplication
            || pendingOptions.enableNutrients != activeOptions.enableNutrients
            || pendingOptions.nutrientDiffusionRate != activeOptions.nutrientDiffusionRate
            || pendingOptions.nutrientAbsorptionRate != activeOptions.nutrientAbsorptionRate
            || pendingOptions.nutrientRegenerationRate != activeOptions.nutrientRegenerationRate
            || pendingOptions.soilLayerHeight != activeOptions.soilLayerHeight
            || pendingOptions.maxNutrient != activeOptions.maxNutrient;
    }
    
    /**
     * @brief Apply pending options as active (call after reset).
     */
    void applyPendingOptions() {
        activeOptions = pendingOptions;
    }
    
    /**
     * @brief Consume the reset flag and return its value.
     * @return true if reset was requested, false otherwise
     */
    bool consumeResetFlag() {
        bool wasRequested = shouldReset;
        shouldReset = false;
        return wasRequested;
    }
};
