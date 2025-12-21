#include <gtest/gtest.h>
#include <cmath>

#include "SimulationTestHelper.h"
#include "simulation/GridTopology.h"
#include "simulation/cpu/LightComputation.h"

class LightComputationTest : public ::testing::Test {
protected:
    static constexpr int WIDTH = 5;
    static constexpr int HEIGHT = 5;
    
    GridTopology topology{WIDTH, HEIGHT};
    SimulationTestHelper helper{topology};
    
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
        OffsetCoord coord{col, row};
        if (topology.isValid(coord)) {
            EXPECT_FLOAT_EQ(helper.getLight(coord), 1.0f)
                << "Air at row " << row << " should have full light";
        }
    }
}

TEST_F(LightComputationTest, TopRowReceivesFullLight) {
    // Place a plant at the top
    const int col = 2;
    const int topRow = topology.topRow();
    OffsetCoord coord{col, topRow};

    helper.setCellType(coord, CellState::Type::Cell);
    
    computeLight();
    
    // Top row always receives full light (absorption affects cells below)
    EXPECT_FLOAT_EQ(helper.getLight(coord), helper.options.lightTopIntensity);
}

TEST_F(LightComputationTest, PlantCellAbsorbsLightForCellsBelow) {
    const int col = 2;
    const int topRow = topology.topRow();
    OffsetCoord coord{col, topRow};
    
    // Place a single plant cell at the top
    helper.setCellType(coord, CellState::Type::Cell);
    
    computeLight();
    
    // Top row: full light
    EXPECT_FLOAT_EQ(helper.getLight(coord), 1.0f);
    
    // Row below attenuated by plant
    OffsetCoord belowCoord{col, topRow - 1};
    float expectedLight = transmittance();
    EXPECT_FLOAT_EQ(helper.getLight(belowCoord), expectedLight)
        << "Cell below plant should receive attenuated light";
}

TEST_F(LightComputationTest, SoilStronglyAbsorbsLight) {
    const int col = 2;
    const int topRow = topology.topRow();
    OffsetCoord coord{col, topRow};
    OffsetCoord belowCoord{col, topRow - 1};
    
    // Place soil at the top
    helper.setCellType(coord, CellState::Type::Soil);
    
    computeLight();
    
    // Top row: full light
    EXPECT_FLOAT_EQ(helper.getLight(coord), 1.0f);
    
    // Row below: almost no light (soil absorbs ~95%)
    float expectedLight = 1.0f - helper.options.soilLightAbsorb;
    EXPECT_FLOAT_EQ(helper.getLight(belowCoord), expectedLight);
    EXPECT_LT(helper.getLight(belowCoord), 0.1f);
}

TEST_F(LightComputationTest, MultiplePlantLayersCompoundAttenuation) {
    const int col = 2;
    
    // Stack of 3 plants at the top of the grid
    const int topRow = topology.topRow();
    OffsetCoord topCoord{col, topRow};
    OffsetCoord secondCoord{col, topRow - 1};
    OffsetCoord thirdCoord{col, topRow - 2};
    OffsetCoord fourthCoord{col, topRow - 3};
    
    helper.setCellType(topCoord, CellState::Type::Cell);
    helper.setCellType(secondCoord, CellState::Type::Cell);
    helper.setCellType(thirdCoord, CellState::Type::Cell);
    
    computeLight();
    
    const float t = transmittance();
    
    // Top plant: full light
    EXPECT_FLOAT_EQ(helper.getLight(topCoord), 1.0f);
    
    // Second plant: attenuated once
    EXPECT_FLOAT_EQ(helper.getLight(secondCoord), t);
    
    // Third plant: attenuated twice
    EXPECT_FLOAT_EQ(helper.getLight(thirdCoord), t * t);
    
    // Cell below all plants: attenuated three times
    EXPECT_FLOAT_EQ(helper.getLight(fourthCoord), t * t * t);
}

TEST_F(LightComputationTest, ColumnsAreIndependent) {
    const int topRow = topology.topRow();
    
    // Plant in column 1 only
    OffsetCoord plantCoord{1, topRow};
    helper.setCellType(plantCoord, CellState::Type::Cell);
    
    computeLight();
    
    // Column 1: attenuated below the plant
    OffsetCoord col1Below{1, topRow - 1};
    EXPECT_FLOAT_EQ(helper.getLight(col1Below), transmittance());
    
    // Column 2: should be unaffected (full light)
    OffsetCoord col2Below{2, topRow - 1};
    EXPECT_FLOAT_EQ(helper.getLight(col2Below), 1.0f);
    
    // Column 0: should be unaffected (full light)
    OffsetCoord col0Below{0, topRow - 1};
    if (topology.isValid(col0Below)) {
        EXPECT_FLOAT_EQ(helper.getLight(col0Below), 1.0f);
    }
}

TEST_F(LightComputationTest, DeadCellsAbsorbLight) {
    const int col = 2;
    const int topRow = topology.topRow();
    
    OffsetCoord deadCoord{col, topRow};
    OffsetCoord belowCoord{col, topRow - 1};
    
    helper.setCellType(deadCoord, CellState::Type::Dead);
    
    computeLight();
    
    float expectedLight = 1.0f - helper.options.deadLightAbsorb;
    EXPECT_FLOAT_EQ(helper.getLight(belowCoord), expectedLight);
}

TEST_F(LightComputationTest, AirGapDoesNotAbsorbLight) {
    const int col = 2;
    const int topRow = topology.topRow();
    
    // Plant, then air gap, then more plant
    OffsetCoord topPlantCoord{col, topRow};
    OffsetCoord airGapCoord{col, topRow - 1};
    OffsetCoord secondPlantCoord{col, topRow - 2};
    OffsetCoord belowSecondCoord{col, topRow - 3};
    
    helper.setCellType(topPlantCoord, CellState::Type::Cell);
    // airGapCoord is Air (default)
    helper.setCellType(secondPlantCoord, CellState::Type::Cell);
    
    computeLight();
    
    const float t = transmittance();
    
    // After first plant (air gap)
    EXPECT_FLOAT_EQ(helper.getLight(airGapCoord), t);
    
    // Second plant receives same light as air gap (air doesn't absorb)
    EXPECT_FLOAT_EQ(helper.getLight(secondPlantCoord), t);
    
    // Below second plant: attenuated again
    EXPECT_FLOAT_EQ(helper.getLight(belowSecondCoord), t * t);
}


// =============================================================================
// Configuration Tests
// =============================================================================

TEST_F(LightComputationTest, TopIntensityParameterIsRespected) {
    helper.options.lightTopIntensity = 0.5f;
    
    computeLight();
    
    OffsetCoord coordTop{2, topology.topRow()};
    OffsetCoord coordBottom{2, topology.bottomRow()};

    EXPECT_FLOAT_EQ(helper.getLight(coordTop), 0.5f);
    EXPECT_FLOAT_EQ(helper.getLight(coordBottom), 0.5f);
}

TEST_F(LightComputationTest, LightCanReachNearZeroWithManyLayers) {
    const int col = 2;
    
    // Fill column with plants from top to bottom
    helper.fillColumn(col, CellState::Type::Cell);
    
    computeLight();
    
    // Bottom row should have very little light
    float expected = std::pow(transmittance(), HEIGHT - 1);
    OffsetCoord coordBottom{col, topology.bottomRow()};
    EXPECT_NEAR(helper.getLight(coordBottom), expected, 0.001f);
}

TEST_F(LightComputationTest, OverwritesPreviousLightValues) {
    OffsetCoord coord{2, 2};
    
    // Set some arbitrary light value
    helper.setLight(coord, 999.0f);
    
    computeLight();
    
    // Should be overwritten with correct value
    EXPECT_FLOAT_EQ(helper.getLight(coord), 1.0f);
}

TEST_F(LightComputationTest, SingleRowGridHasFullLight) {
    GridTopology singleRowTopology{5, 1};
    SimulationTestHelper helper{singleRowTopology};
    helper.options.lightTopIntensity = 1.0f;
    
    LightComputation::compute(helper.state, helper.options);
    
    // All cells should have top light intensity
    for (int col = 0; col < 5; ++col) {
        OffsetCoord coord{col, 0};
        if (singleRowTopology.isValid(coord)) {
            EXPECT_FLOAT_EQ(helper.getLight(coord), 1.0f);
        }
    }
}

// =============================================================================
// Realistic Scenario Tests
// =============================================================================

TEST_F(LightComputationTest, CanopyCreatesVerticalLightGradient) {
    const int col = 2;
    OffsetCoord topCoord{col, topology.topRow()};
    OffsetCoord secondCoord{col, topology.topRow() - 1};
    OffsetCoord thirdCoord{col, topology.topRow() - 2};
    
    // Create a canopy of plants at the top 3 rows
    helper.setCellType(topCoord, CellState::Type::Cell);
    helper.setCellType(secondCoord, CellState::Type::Cell);
    helper.setCellType(thirdCoord, CellState::Type::Cell);
    
    computeLight();
    
    // Light should strictly decrease going down through the canopy
    float lightAbove = helper.getLight(topCoord);
    for (int row = topology.topRow() - 1; row >= topology.topRow() - 2; --row) {
        OffsetCoord coord{col, row};
        float lightHere = helper.getLight(coord);
        EXPECT_LT(lightHere, lightAbove) 
            << "Light should decrease at row " << row;
        lightAbove = lightHere;
    }
}
