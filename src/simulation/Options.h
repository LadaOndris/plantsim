
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
    
    // Light computation parameters
    float lightTopIntensity = 1.0f;        // Light intensity at top of grid
    float plantLightAbsorb = 0.45f;        // Fraction absorbed by plant cells
    float deadLightAbsorb = 0.15f;         // Fraction absorbed by dead cells (future)
    float soilLightAbsorb = 0.95f;         // Fraction absorbed by soil
    
    // Photosynthesis parameters
    float photoMaxRate = 0.08f;            // Max sugar production per tick
    float lightHalfSat = 0.4f;             // Half-saturation for light
    float waterHalfSat = 0.2f;             // Half-saturation for water
    float waterPerSugar = 1.0f;            // Water consumed per unit sugar produced
    
    // Time step for physics
    float dt = 1.0f;
};
