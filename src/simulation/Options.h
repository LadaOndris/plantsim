
#pragma once

struct Options {
    bool enableResourceTransfer = false;
    bool enableCellMultiplication = false;
    bool enableSoilSystem = false;
    bool enableMaintenanceAndDeath = false;
    bool enableDeadDecay = false;
    
    // Soil layer configuration
    int soilLayerHeight = 20;                  // Bottom rows that act as soil
    
    // Soil resource targets (equilibrium values for regeneration)
    float soilWaterTarget = 1.0f;
    float soilMineralTarget = 1.0f;
    
    // Soil regeneration rates (toward target, per tick)
    float soilWaterRegenRate = 0.01f;
    float soilMineralRegenRate = 0.0005f;
    
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
    float photoMaxRate = 0.1f;             // Max sugar production per tick
    float lightHalfSat = 0.4f;             // Half-saturation for light
    float waterHalfSat = 0.2f;             // Half-saturation for water
    float waterPerSugar = 1.0f;            // Water consumed per unit sugar produced
    
    // Maintenance costs (per plant tile per tick)
    float sugarMaintCost = 0.002f;          // Sugar consumed for maintenance
    float waterMaintCost = 0.001f;          // Base water consumed for maintenance
    float waterLightLoss = 0.01f;           // Extra water cost scaled by light (transpiration)
    
    // Health damage from deficits
    float sugarDeficitDamage = 0.01f;       // Health damage per unit sugar deficit
    float waterDeficitDamage = 0.01f;       // Health damage per unit water deficit
    float healthRegenRate = 0.02f;          // Health regeneration per tick when resources are sufficient
    
    // Dead matter decay (recycling into soil)
    float deadDecayRate = 0.02f;           // Fraction per tick of dead stores released
    float deadToSoilBias = 1.0f;           // How strongly dead returns minerals to soil
    
    // Time step for physics
    float dt = 1.0f;

    // Reproduction
    float reproductionThreshold = 1.0f;
    float reproductionCost = 1.0f;
    float childInitialResources = 0.0f;
    float childInitialWater = 0.0f;
    float childInitialHealth = 1.0f;

    bool operator==(const Options&) const = default;
};

inline constexpr Options DefaultOptions{};
