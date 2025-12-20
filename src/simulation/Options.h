
# pragma once

struct Options {
    bool enableResourceTransfer = false;
    bool enableCellMultiplication = false;
    bool enableNutrients = false;
    
    float nutrientDiffusionRate = 0.1f;        // Rate of diffusion between cells
    float nutrientAbsorptionRate = 0.5f;       // Max nutrients a cell can absorb per tick
    float nutrientRegenerationRate = 0.2f;     // Rate of regeneration in soil layer
    int soilLayerHeight = 5;                   // Bottom rows that act as soil (nutrient sources)
    float maxNutrient = 100.0f;                // Maximum nutrient concentration
};
