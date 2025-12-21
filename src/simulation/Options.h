
#pragma once

struct Options {
    bool enableResourceTransfer = false;
    bool enableCellMultiplication = false;
    bool enableSoilSystem = false;
    
    // Soil layer configuration
    int soilLayerHeight = 20;                  // Bottom rows that act as soil
    
    // Soil resource targets (equilibrium values for regeneration)
    float soilWaterTarget = 1.0f;
    float soilMineralTarget = 1.0f;
    
    // Soil regeneration rates (toward target, per tick)
    float soilWaterRegenRate = 0.02f;
    float soilMineralRegenRate = 0.005f;
    
    // Soil diffusion rates (state-dependent, only in soil)
    float soilWaterDiffusivity = 0.18f;
    float soilMineralDiffusivity = 0.10f;
    
    // Uptake from soil into plants (max pulled per edge per tick)
    float waterUptakeRate = 0.08f;
    float mineralUptakeRate = 0.04f;
    
    // Internal plant transport rates (diffusion between plant cells)
    float sugarTransportRate = 0.18f;
    float waterTransportRate = 0.10f;
    float mineralTransportRate = 0.08f;
    
    // Time step for physics
    float dt = 1.0f;
};
