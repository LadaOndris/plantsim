#pragma once

/**
 * @brief Options controlling visualization layer overlays.
 */
struct RenderingOptions {
    // Layer visibility toggles
    bool showResources = true;
    bool showCellTypes = true;
    bool showNutrients = false;
    
    // Layer opacity (0.0 = transparent, 1.0 = opaque)
    float resourcesOpacity = 1.0f;
    float cellTypesOpacity = 0.5f;
    float nutrientsOpacity = 0.7f;
};

