#pragma once

/**
 * @brief Options controlling visualization layer overlays.
 */
struct RenderingOptions {
    // Layer visibility toggles
    bool showSugar = true;
    bool showCellTypes = true;
    bool showWater = true;
    bool showMineral = true;
    
    // Layer opacity (0.0 = transparent, 1.0 = opaque)
    float sugarOpacity = 1.0f;
    float cellTypesOpacity = 0.5f;
    float waterOpacity = 0.6f;
    float mineralOpacity = 0.6f;
};
