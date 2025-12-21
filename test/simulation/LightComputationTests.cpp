#include <gtest/gtest.h>
#include <cmath>

#include "SimulationTestHelper.h"
#include "simulation/cpu/LightComputation.h"

class LightComputationTest : public ::testing::Test {
protected:
    static constexpr int WIDTH = 5;
    static constexpr int HEIGHT = 5;
    
    SimulationTestHelper helper{WIDTH, HEIGHT};
    
    void SetUp() override {
        helper.options.lightTopIntensity = 1.0f;
        helper.options.plantLightAbsorb = 0.45f;
        helper.options.soilLightAbsorb = 0.95f;
        helper.options.deadLightAbsorb = 0.15f;
    }
    
    void computeLight() {
        LightComputation::compute(helper.state, helper.options);
    }
    
    float transmittance() const {
        return 1.0f - helper.options.plantLightAbsorb;
    }
};

// =============================================================================
// Basic Light Propagation Tests
// =============================================================================

TEST_F(LightComputationTest, AirColumnHasFullLightThroughout) {
    // All cells are Air by default
    computeLight();
    
    const int col = 2;
    for (int row = 0; row < HEIGHT; ++row) {
        if (helper.isValid(col, row)) {
            EXPECT_FLOAT_EQ(helper.getLight(col, row), 1.0f)
                << "Air at row " << row << " should have full light";
        }
    }
}

TEST_F(LightComputationTest, TopRowReceivesFullLight) {
    // Place a plant at the top
    const int col = 2;
    const int topRow = helper.topRow();
    helper.setCellType(col, topRow, CellState::Type::Cell);
    
    computeLight();
    
    // Top row always receives full light (absorption affects cells below)
    EXPECT_FLOAT_EQ(helper.getLight(col, topRow), helper.options.lightTopIntensity);
}

TEST_F(LightComputationTest, PlantCellAbsorbsLightForCellsBelow) {
    const int col = 2;
    const int topRow = helper.topRow();
    
    // Place a single plant cell at the top
    helper.setCellType(col, topRow, CellState::Type::Cell);
    
    computeLight();
    
    // Top row: full light
    EXPECT_FLOAT_EQ(helper.getLight(col, topRow), 1.0f);
    
    // Row below (topRow - 1): attenuated by plant
    float expectedLight = transmittance();
    EXPECT_FLOAT_EQ(helper.getLight(col, topRow - 1), expectedLight)
        << "Cell below plant should receive attenuated light";
}

TEST_F(LightComputationTest, SoilStronglyAbsorbsLight) {
    const int col = 2;
    const int topRow = helper.topRow();
    
    // Place soil at the top
    helper.setCellType(col, topRow, CellState::Type::Soil);
    
    computeLight();
    
    // Top row: full light
    EXPECT_FLOAT_EQ(helper.getLight(col, topRow), 1.0f);
    
    // Row below: almost no light (soil absorbs ~95%)
    float expectedLight = 1.0f - helper.options.soilLightAbsorb;
    EXPECT_FLOAT_EQ(helper.getLight(col, topRow - 1), expectedLight);
    EXPECT_LT(helper.getLight(col, topRow - 1), 0.1f);
}

TEST_F(LightComputationTest, MultiplePlantLayersCompoundAttenuation) {
    const int col = 2;
    
    // Stack of 3 plants at the top of the grid
    const int topRow = helper.topRow();
    helper.setCellType(col, topRow, CellState::Type::Cell);
    helper.setCellType(col, topRow - 1, CellState::Type::Cell);
    helper.setCellType(col, topRow - 2, CellState::Type::Cell);
    
    computeLight();
    
    const float t = transmittance();
    
    // Top plant: full light
    EXPECT_FLOAT_EQ(helper.getLight(col, topRow), 1.0f);
    
    // Second plant: attenuated once
    EXPECT_FLOAT_EQ(helper.getLight(col, topRow - 1), t);
    
    // Third plant: attenuated twice
    EXPECT_FLOAT_EQ(helper.getLight(col, topRow - 2), t * t);
    
    // Cell below all plants: attenuated three times
    EXPECT_FLOAT_EQ(helper.getLight(col, topRow - 3), t * t * t);
}

TEST_F(LightComputationTest, ColumnsAreIndependent) {
    const int topRow = helper.topRow();
    
    // Plant in column 1 only
    helper.setCellType(1, topRow, CellState::Type::Cell);
    
    computeLight();
    
    // Column 1: attenuated below the plant
    EXPECT_FLOAT_EQ(helper.getLight(1, topRow - 1), transmittance());
    
    // Column 2: should be unaffected (full light)
    EXPECT_FLOAT_EQ(helper.getLight(2, topRow - 1), 1.0f);
    
    // Column 0: should be unaffected (full light)
    if (helper.isValid(0, topRow - 1)) {
        EXPECT_FLOAT_EQ(helper.getLight(0, topRow - 1), 1.0f);
    }
}

TEST_F(LightComputationTest, DeadCellsAbsorbLight) {
    const int col = 2;
    const int topRow = helper.topRow();
    
    helper.setCellType(col, topRow, CellState::Type::Dead);
    
    computeLight();
    
    float expectedLight = 1.0f - helper.options.deadLightAbsorb;
    EXPECT_FLOAT_EQ(helper.getLight(col, topRow - 1), expectedLight);
}

TEST_F(LightComputationTest, AirGapDoesNotAbsorbLight) {
    const int col = 2;
    const int topRow = helper.topRow();
    
    // Plant, then air gap, then more plant
    helper.setCellType(col, topRow, CellState::Type::Cell);
    // topRow - 1 is Air (default)
    helper.setCellType(col, topRow - 2, CellState::Type::Cell);
    
    computeLight();
    
    const float t = transmittance();
    
    // After first plant (air gap)
    EXPECT_FLOAT_EQ(helper.getLight(col, topRow - 1), t);
    
    // Second plant receives same light as air gap (air doesn't absorb)
    EXPECT_FLOAT_EQ(helper.getLight(col, topRow - 2), t);
    
    // Below second plant: attenuated again
    EXPECT_FLOAT_EQ(helper.getLight(col, topRow - 3), t * t);
}

// =============================================================================
// Configuration Tests
// =============================================================================

TEST_F(LightComputationTest, TopIntensityParameterIsRespected) {
    helper.options.lightTopIntensity = 0.5f;
    
    computeLight();
    
    const int col = 2;
    EXPECT_FLOAT_EQ(helper.getLight(col, helper.topRow()), 0.5f);
    EXPECT_FLOAT_EQ(helper.getLight(col, helper.bottomRow()), 0.5f);
}

TEST_F(LightComputationTest, LightCanReachNearZeroWithManyLayers) {
    const int col = 2;
    
    // Fill column with plants from top to bottom
    helper.fillColumn(col, CellState::Type::Cell);
    
    computeLight();
    
    // Bottom row should have very little light
    float expected = std::pow(transmittance(), HEIGHT - 1);
    EXPECT_NEAR(helper.getLight(col, helper.bottomRow()), expected, 0.001f);
}

TEST_F(LightComputationTest, OverwritesPreviousLightValues) {
    const int col = 2;
    const int row = 2;
    
    // Set some arbitrary light value
    helper.setLight(col, row, 999.0f);
    
    computeLight();
    
    // Should be overwritten with correct value
    EXPECT_FLOAT_EQ(helper.getLight(col, row), 1.0f);
}

TEST_F(LightComputationTest, SingleRowGridHasFullLight) {
    SimulationTestHelper singleRow{5, 1};
    singleRow.options.lightTopIntensity = 1.0f;
    
    LightComputation::compute(singleRow.state, singleRow.options);
    
    // All cells should have top light intensity
    for (int col = 0; col < 5; ++col) {
        if (singleRow.isValid(col, 0)) {
            EXPECT_FLOAT_EQ(singleRow.getLight(col, 0), 1.0f);
        }
    }
}

// =============================================================================
// Realistic Scenario Tests
// =============================================================================

TEST_F(LightComputationTest, CanopyCreatesVerticalLightGradient) {
    const int col = 2;
    
    // Create a canopy of plants at the top 3 rows
    helper.setCellType(col, helper.topRow(), CellState::Type::Cell);
    helper.setCellType(col, helper.topRow() - 1, CellState::Type::Cell);
    helper.setCellType(col, helper.topRow() - 2, CellState::Type::Cell);
    
    computeLight();
    
    // Light should strictly decrease going down through the canopy
    float lightAbove = helper.getLight(col, helper.topRow());
    for (int row = helper.topRow() - 1; row >= helper.topRow() - 2; --row) {
        float lightHere = helper.getLight(col, row);
        EXPECT_LT(lightHere, lightAbove) 
            << "Light should decrease at row " << row;
        lightAbove = lightHere;
    }
}

TEST_F(LightComputationTest, SoilAtBottomBlocksRemainingLight) {
    const int col = 2;
    
    // Plant at top, soil at bottom
    helper.setCellType(col, helper.topRow(), CellState::Type::Cell);
    helper.setCellType(col, helper.bottomRow(), CellState::Type::Soil);
    helper.setCellType(col, helper.bottomRow() + 1, CellState::Type::Soil);
    
    computeLight();
    
    // Light at bottom should be heavily attenuated
    float lightAtBottom = helper.getLight(col, helper.bottomRow());
    EXPECT_LT(lightAtBottom, 0.1f);
}
