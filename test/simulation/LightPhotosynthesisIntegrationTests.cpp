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
    void placePlant(OffsetCoord coord, float water = 1.0f) {
        helper.setCellType(coord, CellState::Type::Cell);
        helper.setPlantWater(coord, water);
    }
};

// =============================================================================
// Basic Integration Tests
// =============================================================================

TEST_F(LightPhotosynthesisIntegrationTest, SinglePlantAtTopProducesSugar) {
    OffsetCoord coord{2, helper.topRow()};
    
    placePlant(coord);
    
    runSimulationStep();
    
    // Should receive full light and produce sugar
    EXPECT_FLOAT_EQ(helper.getLight(coord), 1.0f);
    EXPECT_GT(helper.getPlantSugar(coord), 0.0f);
}

TEST_F(LightPhotosynthesisIntegrationTest, PlantWithoutWaterProducesNoSugar) {
    OffsetCoord coord{2, helper.topRow()};
    
    placePlant(coord, 0.0f);
    
    runSimulationStep();
    
    // Has light but no water
    EXPECT_FLOAT_EQ(helper.getLight(coord), 1.0f);
    EXPECT_FLOAT_EQ(helper.getPlantSugar(coord), 0.0f);
}

// =============================================================================
// Canopy Shading Tests
// =============================================================================

struct CanopyDepthParam {
    int numLayers;
    const char* description;
};

class CanopyDepthTest : public LightPhotosynthesisIntegrationTest,
                         public ::testing::WithParamInterface<CanopyDepthParam> {};

TEST_P(CanopyDepthTest, CanopyCreatesLightAndSugarGradient) {
    const auto param = GetParam();
    const int col = 2;
    const int topRow = helper.topRow();
    
    // Create vertical stack of plants
    std::vector<OffsetCoord> coords;
    for (int i = 0; i < param.numLayers; ++i) {
        coords.push_back({col, topRow - i});
        placePlant(coords.back());
    }
    
    runSimulationStep();
    
    // Verify light gradient
    const float t = transmittance();
    for (int i = 0; i < param.numLayers; ++i) {
        float expectedLight = std::pow(t, i);
        EXPECT_FLOAT_EQ(helper.getLight(coords[i]), expectedLight)
            << "Light at layer " << i << " with " << param.description;
    }
    
    // Verify sugar production decreases with depth
    for (int i = 1; i < param.numLayers; ++i) {
        EXPECT_GT(helper.getPlantSugar(coords[i-1]), helper.getPlantSugar(coords[i]))
            << "Sugar at layer " << i-1 << " should be > layer " << i;
        EXPECT_GT(helper.getPlantSugar(coords[i]), 0.0f)
            << "Layer " << i << " should still produce some sugar";
    }
}

INSTANTIATE_TEST_SUITE_P(
    LayerDepths,
    CanopyDepthTest,
    ::testing::Values(
        CanopyDepthParam{2, "2-layer canopy"},
        CanopyDepthParam{3, "3-layer canopy"},
        CanopyDepthParam{4, "4-layer canopy"}
    )
);

TEST_F(LightPhotosynthesisIntegrationTest, CanopyShadesLowerPlants) {
    const int col = 2;
    const int topRow = helper.topRow();
    OffsetCoord coordTop{col, topRow};
    OffsetCoord coordLower{col, topRow - 1};
    
    // Vertical stack: top plant shades lower plant
    placePlant(coordTop);
    placePlant(coordLower);
    
    runSimulationStep();
    
    float topSugar = helper.getPlantSugar(coordTop);
    float lowerSugar = helper.getPlantSugar(coordLower);
    
    // Top plant produces more than shaded lower plant
    EXPECT_GT(topSugar, lowerSugar);
    
    // Both produce some sugar
    EXPECT_GT(topSugar, 0.0f);
    EXPECT_GT(lowerSugar, 0.0f);
}

TEST_F(LightPhotosynthesisIntegrationTest, AdjacentColumnsAreIndependent) {
    const int topRow = helper.topRow();
    OffsetCoord coord1_top{1, topRow};
    OffsetCoord coord1_mid{1, topRow - 1};
    OffsetCoord coord1_bot{1, topRow - 2};
    OffsetCoord coord3_top{3, topRow};
    
    // Dense canopy in column 1, single plant in column 3
    placePlant(coord1_top);
    placePlant(coord1_mid);
    placePlant(coord1_bot);
    
    placePlant(coord3_top);
    
    runSimulationStep();
    
    // Column 3's single plant should get full light
    EXPECT_FLOAT_EQ(helper.getLight(coord3_top), 1.0f);
    
    // Column 1's bottom plant gets much less light
    const float t = transmittance();
    EXPECT_FLOAT_EQ(helper.getLight(coord1_bot), t * t);
    
    // Sugar production reflects light availability
    EXPECT_GT(helper.getPlantSugar(coord3_top), helper.getPlantSugar(coord1_bot));
}

// =============================================================================
// Air Gaps and Mixed Layouts
// =============================================================================

TEST_F(LightPhotosynthesisIntegrationTest, AirGapAllowsLightToLowerPlant) {
    const int col = 2;
    const int topRow = helper.topRow();
    OffsetCoord coordTop{col, topRow};
    OffsetCoord coordLower{col, topRow - 2};
    
    // Plant at top, air gap, plant below
    placePlant(coordTop);
    // topRow - 1 is Air
    placePlant(coordLower);
    
    runSimulationStep();
    
    const float t = transmittance();
    
    // Lower plant receives attenuated light (only from top plant)
    EXPECT_FLOAT_EQ(helper.getLight(coordLower), t);
    
    // Both produce sugar, top produces more
    EXPECT_GT(helper.getPlantSugar(coordTop), helper.getPlantSugar(coordLower));
}

TEST_F(LightPhotosynthesisIntegrationTest, SoilBlocksLightCompletely) {
    const int col = 2;
    const int topRow = helper.topRow();
    OffsetCoord coordTop{col, topRow};
    OffsetCoord coordBelow{col, topRow - 1};
    
    // Soil at top blocks almost all light
    helper.setCellType(coordTop, CellState::Type::Soil);
    placePlant(coordBelow);  // Plant below soil
    
    runSimulationStep();
    
    // Very little light reaches the plant
    EXPECT_LT(helper.getLight(coordBelow), 0.1f);
    
    // Very little sugar produced
    EXPECT_LT(helper.getPlantSugar(coordBelow), 0.1f);
}

// =============================================================================
// Multiple Simulation Steps
// =============================================================================

struct AccumulationParam {
    int numSteps;
    float tolerance;
};

class AccumulationLinearityTest : public LightPhotosynthesisIntegrationTest,
                                   public ::testing::WithParamInterface<AccumulationParam> {};

TEST_P(AccumulationLinearityTest, SugarAccumulationIsLinear) {
    const auto param = GetParam();
    OffsetCoord coord{2, helper.topRow()};
    
    placePlant(coord, 10.0f);  // Lots of water to avoid depletion
    
    runSimulationStep();
    float perStepIncrement = helper.getPlantSugar(coord);
    
    // Run remaining steps
    for (int i = 1; i < param.numSteps; ++i) {
        runSimulationStep();
    }
    
    float total = helper.getPlantSugar(coord);
    float expected = param.numSteps * perStepIncrement;
    
    EXPECT_NEAR(total, expected, param.tolerance)
        << "After " << param.numSteps << " steps, accumulation should be linear";
}

INSTANTIATE_TEST_SUITE_P(
    MultipleSteps,
    AccumulationLinearityTest,
    ::testing::Values(
        AccumulationParam{3, 0.01f},
        AccumulationParam{5, 0.01f},
        AccumulationParam{10, 0.02f}
    )
);

TEST_F(LightPhotosynthesisIntegrationTest, SugarAccumulatesOverTime) {
    OffsetCoord coord{2, helper.topRow()};
    
    placePlant(coord, 10.0f);  // Lots of water to avoid depletion
    
    std::vector<float> sugarLevels;
    
    for (int step = 0; step < 5; ++step) {
        runSimulationStep();
        sugarLevels.push_back(helper.getPlantSugar(coord));
    }
    
    // Sugar should strictly increase each step
    for (size_t i = 1; i < sugarLevels.size(); ++i) {
        EXPECT_GT(sugarLevels[i], sugarLevels[i-1])
            << "Sugar should increase at step " << i;
    }
}

// =============================================================================
// Realistic Scenarios
// =============================================================================

TEST_F(LightPhotosynthesisIntegrationTest, RealisticPlantAboveSoil) {
    const int col = 2;
    OffsetCoord coordSoil0{col, 0};
    OffsetCoord coordSoil1{col, 1};
    OffsetCoord coordPlant0{col, 2};
    OffsetCoord coordPlant1{col, 3};
    OffsetCoord coordPlant2{col, 4};
    
    // Soil at bottom 2 rows
    helper.setCellType(coordSoil0, CellState::Type::Soil);
    helper.setCellType(coordSoil1, CellState::Type::Soil);
    
    // Plant growing from soil (rows 2-4)
    placePlant(coordPlant0);
    placePlant(coordPlant1);
    placePlant(coordPlant2);
    
    runSimulationStep();
    
    // Top of plant (row 4) gets more light than bottom (row 2)
    EXPECT_GT(helper.getLight(coordPlant2), helper.getLight(coordPlant0));
    
    // Top of plant produces more sugar
    EXPECT_GT(helper.getPlantSugar(coordPlant2), helper.getPlantSugar(coordPlant0));
    
    // All plant cells produce some sugar
    EXPECT_GT(helper.getPlantSugar(coordPlant0), 0.0f);
    EXPECT_GT(helper.getPlantSugar(coordPlant1), 0.0f);
    EXPECT_GT(helper.getPlantSugar(coordPlant2), 0.0f);
}

TEST_F(LightPhotosynthesisIntegrationTest, ZeroLightIntensityProducesNoSugar) {
    OffsetCoord coord{2, helper.topRow()};
    
    helper.options.lightTopIntensity = 0.0f;
    placePlant(coord);
    
    runSimulationStep();
    
    EXPECT_FLOAT_EQ(helper.getLight(coord), 0.0f);
    EXPECT_FLOAT_EQ(helper.getPlantSugar(coord), 0.0f);
}
