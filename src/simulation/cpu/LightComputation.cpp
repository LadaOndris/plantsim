#include "simulation/cpu/LightComputation.h"
#include "simulation/GridTopology.h"

void LightComputation::compute(State& state, const Options& options) {
    const int logicalWidth = state.width;
    const int logicalHeight = state.height;
    
    GridTopology topology(logicalWidth, logicalHeight);
    StorageCoord storageDim = topology.storageDim;
    
    // Resize light vector if needed
    const size_t totalStorageCells = static_cast<size_t>(storageDim.x) * storageDim.y;
    if (state.light.size() != totalStorageCells) {
        state.light.resize(totalStorageCells, 0.0f);
    }
    
    // Process each logical column independently (light propagates vertically)
    // Coordinate system: row 0 is BOTTOM, row (height-1) is TOP (sky)
    // Light enters from top and propagates downward through offset columns
    for (int col = 0; col < logicalWidth; ++col) {
        float intensity = options.lightTopIntensity;
        
        for (int row = logicalHeight - 1; row >= 0; --row) {
            // Convert offset coordinates to storage index
            OffsetCoord offset{col, row};
            int idx = topology.toStorageIndex(offset);
            
            state.light[idx] = intensity;
            
            const auto cellType = static_cast<CellState::Type>(state.cellTypes[idx]);
            
            switch (cellType) {
                case CellState::Type::Cell:
                    intensity *= (1.0f - options.plantLightAbsorb);
                    break;
                    
                case CellState::Type::Soil:
                    intensity *= (1.0f - options.soilLightAbsorb);
                    break;
                    
                case CellState::Type::Dead:
                    intensity *= (1.0f - options.deadLightAbsorb);
                    break;
                    
                case CellState::Type::Air:
                    // Air does not attenuate light
                    break;
            }
        }
    }
}
