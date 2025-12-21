#include <gtest/gtest.h>

#include "SimulationTestHelper.h"
#include "simulation/cpu/Photosynthesis.h"

class PhotosynthesisTest : public ::testing::Test {
protected:
    static constexpr int WIDTH = 5;
    static constexpr int HEIGHT = 5;
    
    SimulationTestHelper helper{WIDTH, HEIGHT};
    
    void SetUp() override {
        helper.options.dt = 1.0f;
        helper.options.photoMaxRate = 1.0f;
        helper.options.lightHalfSat = 0.5f;
        helper.options.waterHalfSat = 0.2f;
    }
    
    void applyPhotosynthesis() {
        Photosynthesis::apply(helper.state, helper.options);
    }
    
    /**
     * @brief Calculate expected sugar production in the same way as Photosynthesis.cpp
     */
    float expectedSugar(float light, float water) const {
        float lightTerm = light / (light + helper.options.lightHalfSat);
        float waterTerm = water / (water + helper.options.waterHalfSat);
        return helper.options.dt * helper.options.photoMaxRate * lightTerm * waterTerm;
    }
    
    /**
     * @brief Set up a plant cell with specified light and water.
     */
    void setupPlantCell(int col, int row, float light, float water) {
        helper.setCellType(col, row, CellState::Type::Cell);
        helper.setLight(col, row, light);
        helper.setPlantWater(col, row, water);
    }
};

// =============================================================================
// Basic Functionality Tests
// =============================================================================

TEST_F(PhotosynthesisTest, AirCellsDoNotPhotosynthesize) {
    const int col = 2, row = 2;
    
    // Air cell with light and water shouldn't produce sugar
    helper.setLight(col, row, 1.0f);
    helper.setPlantWater(col, row, 1.0f);
    // Cell type is Air by default
    
    applyPhotosynthesis();
    
    EXPECT_FLOAT_EQ(helper.getPlantSugar(col, row), 0.0f);
}

TEST_F(PhotosynthesisTest, PlantCellProducesSugarWithLightAndWater) {
    const int col = 2, row = 2;
    
    setupPlantCell(col, row, 1.0f, 1.0f);
    
    applyPhotosynthesis();
    
    float expected = expectedSugar(1.0f, 1.0f);
    EXPECT_FLOAT_EQ(helper.getPlantSugar(col, row), expected);
    EXPECT_GT(expected, 0.0f);  // Sanity check
}

TEST_F(PhotosynthesisTest, NoPhotosynthesisWithoutLight) {
    const int col = 2, row = 2;
    
    setupPlantCell(col, row, 0.0f, 1.0f);
    
    applyPhotosynthesis();
    
    EXPECT_FLOAT_EQ(helper.getPlantSugar(col, row), 0.0f);
}

TEST_F(PhotosynthesisTest, NoPhotosynthesisWithoutWater) {
    const int col = 2, row = 2;
    
    setupPlantCell(col, row, 1.0f, 0.0f);
    
    applyPhotosynthesis();
    
    EXPECT_FLOAT_EQ(helper.getPlantSugar(col, row), 0.0f);
}

// =============================================================================
// Michaelis-Menten Saturation Tests
// =============================================================================

TEST_F(PhotosynthesisTest, LightSaturationFollowsMichaelisMenten) {
    const int col = 2, row = 2;
    
    // At half-saturation light, light term should be 0.5
    setupPlantCell(col, row, helper.options.lightHalfSat, 1.0f);
    
    applyPhotosynthesis();
    
    float expected = expectedSugar(helper.options.lightHalfSat, 1.0f);
    EXPECT_FLOAT_EQ(helper.getPlantSugar(col, row), expected);
    
    // Verify light term is 0.5 at half-saturation
    float lightTerm = helper.options.lightHalfSat / (helper.options.lightHalfSat + helper.options.lightHalfSat);
    EXPECT_FLOAT_EQ(lightTerm, 0.5f);
}

TEST_F(PhotosynthesisTest, WaterSaturationFollowsMichaelisMenten) {
    const int col = 2, row = 2;
    
    // At half-saturation water, water term should be 0.5
    setupPlantCell(col, row, 1.0f, helper.options.waterHalfSat);
    
    applyPhotosynthesis();
    
    float expected = expectedSugar(1.0f, helper.options.waterHalfSat);
    EXPECT_FLOAT_EQ(helper.getPlantSugar(col, row), expected);
    
    // Verify water term is 0.5 at half-saturation
    float waterTerm = helper.options.waterHalfSat / (helper.options.waterHalfSat + helper.options.waterHalfSat);
    EXPECT_FLOAT_EQ(waterTerm, 0.5f);
}

TEST_F(PhotosynthesisTest, HighResourcesApproachMaxRate) {
    const int col = 2, row = 2;
    
    // Very high light and water should approach max rate
    setupPlantCell(col, row, 100.0f, 100.0f);
    
    applyPhotosynthesis();
    
    // With very high resources, both terms approach 1.0
    // So production should approach dt * maxRate = 1.0
    float sugar = helper.getPlantSugar(col, row);
    EXPECT_NEAR(sugar, helper.options.photoMaxRate * helper.options.dt, 0.02f);
}

// =============================================================================
// Multi-Cell and Accumulation Tests
// =============================================================================

TEST_F(PhotosynthesisTest, MultiplePlantCellsWorkIndependently) {
    // Two plant cells with different light levels
    setupPlantCell(1, 2, 1.0f, 1.0f);
    setupPlantCell(3, 2, 0.5f, 1.0f);
    
    applyPhotosynthesis();
    
    EXPECT_FLOAT_EQ(helper.getPlantSugar(1, 2), expectedSugar(1.0f, 1.0f));
    EXPECT_FLOAT_EQ(helper.getPlantSugar(3, 2), expectedSugar(0.5f, 1.0f));
    EXPECT_GT(helper.getPlantSugar(1, 2), helper.getPlantSugar(3, 2));
}

TEST_F(PhotosynthesisTest, SugarAccumulatesOverMultipleSteps) {
    const int col = 2, row = 2;
    
    setupPlantCell(col, row, 1.0f, 1.0f);
    
    // First step
    applyPhotosynthesis();
    float afterFirst = helper.getPlantSugar(col, row);
    
    // Second step
    applyPhotosynthesis();
    float afterSecond = helper.getPlantSugar(col, row);
    
    EXPECT_FLOAT_EQ(afterSecond, 2.0f * afterFirst);
}

TEST_F(PhotosynthesisTest, TimeStepScalesProduction) {
    const int col = 2, row = 2;
    
    helper.options.dt = 0.5f;
    setupPlantCell(col, row, 1.0f, 1.0f);
    
    applyPhotosynthesis();
    
    // With dt=0.5, production should be half
    float expected = 0.5f * helper.options.photoMaxRate 
                   * (1.0f / (1.0f + helper.options.lightHalfSat))
                   * (1.0f / (1.0f + helper.options.waterHalfSat));
    EXPECT_FLOAT_EQ(helper.getPlantSugar(col, row), expected);
}

// =============================================================================
// Cell Type Filtering Tests
// =============================================================================

TEST_F(PhotosynthesisTest, SoilCellsDoNotPhotosynthesize) {
    const int col = 2, row = 2;
    
    helper.setCellType(col, row, CellState::Type::Soil);
    helper.setLight(col, row, 1.0f);
    helper.setPlantWater(col, row, 1.0f);
    
    applyPhotosynthesis();
    
    EXPECT_FLOAT_EQ(helper.getPlantSugar(col, row), 0.0f);
}

TEST_F(PhotosynthesisTest, DeadCellsDoNotPhotosynthesize) {
    const int col = 2, row = 2;
    
    helper.setCellType(col, row, CellState::Type::Dead);
    helper.setLight(col, row, 1.0f);
    helper.setPlantWater(col, row, 1.0f);
    
    applyPhotosynthesis();
    
    EXPECT_FLOAT_EQ(helper.getPlantSugar(col, row), 0.0f);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(PhotosynthesisTest, VeryLowLightProducesMinimalSugar) {
    const int col = 2, row = 2;
    
    setupPlantCell(col, row, 0.01f, 1.0f);
    
    applyPhotosynthesis();
    
    float sugar = helper.getPlantSugar(col, row);
    EXPECT_GT(sugar, 0.0f);  // Still produces some
    EXPECT_LT(sugar, 0.1f);   // But very little
}

TEST_F(PhotosynthesisTest, VeryLowWaterProducesMinimalSugar) {
    const int col = 2, row = 2;
    
    setupPlantCell(col, row, 1.0f, 0.01f);
    
    applyPhotosynthesis();
    
    float sugar = helper.getPlantSugar(col, row);
    EXPECT_GT(sugar, 0.0f);  // Still produces some
    EXPECT_LT(sugar, 0.1f);   // But very little
}
