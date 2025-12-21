#include <gtest/gtest.h>
#include <cmath>

#include "SimulationTestHelper.h"
#include "simulation/cpu/LightComputation.h"
#include "simulation/cpu/Photosynthesis.h"

class LightPhotosynthesisIntegrationTest : public ::testing::Test {
protected:
    static constexpr int WIDTH = 5;
    static constexpr int HEIGHT = 8;
    
    SimulationTestHelper helper{WIDTH, HEIGHT};
    
    void SetUp() override {
        helper.options.lightTopIntensity = 1.0f;
        helper.options.plantLightAbsorb = 0.45f;
        helper.options.soilLightAbsorb = 0.95f;
        helper.options.dt = 1.0f;
        helper.options.photoMaxRate = 1.0f;
        helper.options.lightHalfSat = 0.5f;
        helper.options.waterHalfSat = 0.2f;
    }
    
    void runSimulationStep() {
        LightComputation::compute(helper.state, helper.options);
        Photosynthesis::apply(helper.state, helper.options);
    }
    
    float expectedSugar(float light, float water) const {
        float lightTerm = light / (light + helper.options.lightHalfSat);
        float waterTerm = water / (water + helper.options.waterHalfSat);
        return helper.options.dt * helper.options.photoMaxRate * lightTerm * waterTerm;
    }
    
    float transmittance() const {
        return 1.0f - helper.options.plantLightAbsorb;
    }
    
    /**
     * @brief Set up a plant cell with water (for photosynthesis).
     */
    void placePlant(int col, int row, float water = 1.0f) {
        helper.setCellType(col, row, CellState::Type::Cell);
        helper.setPlantWater(col, row, water);
    }
};

// =============================================================================
// Basic Integration Tests
// =============================================================================

TEST_F(LightPhotosynthesisIntegrationTest, SinglePlantAtTopProducesSugar) {
    const int col = 2;
    const int topRow = helper.topRow();
    
    placePlant(col, topRow);
    
    runSimulationStep();
    
    // Should receive full light and produce sugar
    EXPECT_FLOAT_EQ(helper.getLight(col, topRow), 1.0f);
    EXPECT_GT(helper.getPlantSugar(col, topRow), 0.0f);
}

TEST_F(LightPhotosynthesisIntegrationTest, PlantWithoutWaterProducesNoSugar) {
    const int col = 2;
    const int topRow = helper.topRow();
    
    placePlant(col, topRow, 0.0f);  // No water
    
    runSimulationStep();
    
    // Has light but no water
    EXPECT_FLOAT_EQ(helper.getLight(col, topRow), 1.0f);
    EXPECT_FLOAT_EQ(helper.getPlantSugar(col, topRow), 0.0f);
}

// =============================================================================
// Canopy Shading Tests
// =============================================================================

TEST_F(LightPhotosynthesisIntegrationTest, CanopyShadesLowerPlants) {
    const int col = 2;
    const int topRow = helper.topRow();
    
    // Vertical stack: top plant shades lower plant
    placePlant(col, topRow);
    placePlant(col, topRow - 1);
    
    runSimulationStep();
    
    float topSugar = helper.getPlantSugar(col, topRow);
    float lowerSugar = helper.getPlantSugar(col, topRow - 1);
    
    // Top plant produces more than shaded lower plant
    EXPECT_GT(topSugar, lowerSugar);
    
    // Both produce some sugar
    EXPECT_GT(topSugar, 0.0f);
    EXPECT_GT(lowerSugar, 0.0f);
}

TEST_F(LightPhotosynthesisIntegrationTest, ThreeLayerCanopyCreatesGradient) {
    const int col = 2;
    const int topRow = helper.topRow();
    
    // Stack of 3 plants
    placePlant(col, topRow);
    placePlant(col, topRow - 1);
    placePlant(col, topRow - 2);
    
    runSimulationStep();
    
    float sugar0 = helper.getPlantSugar(col, topRow);
    float sugar1 = helper.getPlantSugar(col, topRow - 1);
    float sugar2 = helper.getPlantSugar(col, topRow - 2);
    
    // Sugar production should strictly decrease with depth
    EXPECT_GT(sugar0, sugar1);
    EXPECT_GT(sugar1, sugar2);
    
    // Verify light gradient matches expectations
    const float t = transmittance();
    EXPECT_FLOAT_EQ(helper.getLight(col, topRow), 1.0f);
    EXPECT_FLOAT_EQ(helper.getLight(col, topRow - 1), t);
    EXPECT_FLOAT_EQ(helper.getLight(col, topRow - 2), t * t);
}

TEST_F(LightPhotosynthesisIntegrationTest, AdjacentColumnsAreIndependent) {
    const int topRow = helper.topRow();
    
    // Dense canopy in column 1, single plant in column 3
    placePlant(1, topRow);
    placePlant(1, topRow - 1);
    placePlant(1, topRow - 2);
    
    placePlant(3, topRow);
    
    runSimulationStep();
    
    // Column 3's single plant should get full light
    EXPECT_FLOAT_EQ(helper.getLight(3, topRow), 1.0f);
    
    // Column 1's bottom plant gets much less light
    const float t = transmittance();
    EXPECT_FLOAT_EQ(helper.getLight(1, topRow - 2), t * t);
    
    // Sugar production reflects light availability
    EXPECT_GT(helper.getPlantSugar(3, topRow), helper.getPlantSugar(1, topRow - 2));
}

// =============================================================================
// Air Gaps and Mixed Layouts
// =============================================================================

TEST_F(LightPhotosynthesisIntegrationTest, AirGapAllowsLightToLowerPlant) {
    const int col = 2;
    const int topRow = helper.topRow();
    
    // Plant at top, air gap, plant below
    placePlant(col, topRow);
    // topRow - 1 is Air
    placePlant(col, topRow - 2);
    
    runSimulationStep();
    
    const float t = transmittance();
    
    // Lower plant receives attenuated light (only from top plant)
    EXPECT_FLOAT_EQ(helper.getLight(col, topRow - 2), t);
    
    // Both produce sugar, top produces more
    EXPECT_GT(helper.getPlantSugar(col, topRow), helper.getPlantSugar(col, topRow - 2));
}

TEST_F(LightPhotosynthesisIntegrationTest, SoilBlocksLightCompletely) {
    const int col = 2;
    const int topRow = helper.topRow();
    
    // Soil at top blocks almost all light
    helper.setCellType(col, topRow, CellState::Type::Soil);
    placePlant(col, topRow - 1);  // Plant below soil
    
    runSimulationStep();
    
    // Very little light reaches the plant
    EXPECT_LT(helper.getLight(col, topRow - 1), 0.1f);
    
    // Very little sugar produced
    EXPECT_LT(helper.getPlantSugar(col, topRow - 1), 0.1f);
}

// =============================================================================
// Multiple Simulation Steps
// =============================================================================

TEST_F(LightPhotosynthesisIntegrationTest, SugarAccumulatesOverTime) {
    const int col = 2;
    const int topRow = helper.topRow();
    
    placePlant(col, topRow, 10.0f);  // Lots of water to avoid depletion
    
    std::vector<float> sugarLevels;
    
    for (int step = 0; step < 5; ++step) {
        runSimulationStep();
        sugarLevels.push_back(helper.getPlantSugar(col, topRow));
    }
    
    // Sugar should strictly increase each step
    for (size_t i = 1; i < sugarLevels.size(); ++i) {
        EXPECT_GT(sugarLevels[i], sugarLevels[i-1])
            << "Sugar should increase at step " << i;
    }
}

TEST_F(LightPhotosynthesisIntegrationTest, AccumulationIsApproximatelyLinear) {
    const int col = 2;
    const int topRow = helper.topRow();
    
    placePlant(col, topRow, 10.0f);  // Lots of water
    
    runSimulationStep();
    float firstIncrement = helper.getPlantSugar(col, topRow);
    
    runSimulationStep();
    runSimulationStep();
    runSimulationStep();
    runSimulationStep();
    float totalAfterFive = helper.getPlantSugar(col, topRow);
    
    // Should be approximately 5x the first increment
    EXPECT_NEAR(totalAfterFive, 5.0f * firstIncrement, 0.01f);
}

// =============================================================================
// Realistic Scenarios
// =============================================================================

TEST_F(LightPhotosynthesisIntegrationTest, RealisticPlantAboveSoil) {
    const int col = 2;
    
    // Soil at bottom 2 rows
    helper.setCellType(col, 0, CellState::Type::Soil);
    helper.setCellType(col, 1, CellState::Type::Soil);
    
    // Plant growing from soil (rows 2-4)
    placePlant(col, 2);
    placePlant(col, 3);
    placePlant(col, 4);
    
    runSimulationStep();
    
    // Top of plant (row 4) gets more light than bottom (row 2)
    EXPECT_GT(helper.getLight(col, 4), helper.getLight(col, 2));
    
    // Top of plant produces more sugar
    EXPECT_GT(helper.getPlantSugar(col, 4), helper.getPlantSugar(col, 2));
    
    // All plant cells produce some sugar
    EXPECT_GT(helper.getPlantSugar(col, 2), 0.0f);
    EXPECT_GT(helper.getPlantSugar(col, 3), 0.0f);
    EXPECT_GT(helper.getPlantSugar(col, 4), 0.0f);
}

TEST_F(LightPhotosynthesisIntegrationTest, ZeroLightIntensityProducesNoSugar) {
    const int col = 2;
    const int topRow = helper.topRow();
    
    helper.options.lightTopIntensity = 0.0f;
    placePlant(col, topRow);
    
    runSimulationStep();
    
    EXPECT_FLOAT_EQ(helper.getLight(col, topRow), 0.0f);
    EXPECT_FLOAT_EQ(helper.getPlantSugar(col, topRow), 0.0f);
}
