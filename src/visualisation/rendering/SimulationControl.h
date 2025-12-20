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
            || pendingOptions.enableSoilSystem != activeOptions.enableSoilSystem
            || pendingOptions.soilLayerHeight != activeOptions.soilLayerHeight
            || pendingOptions.soilWaterTarget != activeOptions.soilWaterTarget
            || pendingOptions.soilMineralTarget != activeOptions.soilMineralTarget
            || pendingOptions.soilWaterRegenRate != activeOptions.soilWaterRegenRate
            || pendingOptions.soilMineralRegenRate != activeOptions.soilMineralRegenRate
            || pendingOptions.soilWaterDiffusivity != activeOptions.soilWaterDiffusivity
            || pendingOptions.soilMineralDiffusivity != activeOptions.soilMineralDiffusivity;
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
