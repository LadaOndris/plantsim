#include <gtest/gtest.h>

#include "SimulationTestHelper.h"
#include "simulation/GridTopology.h"
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
    void setupPlantCell(OffsetCoord coord, float light, float water) {
        helper.setCellType(coord, CellState::Type::Cell);
        helper.setLight(coord, light);
        helper.setPlantWater(coord, water);
    }
};

// =============================================================================
// Basic Functionality Tests
// =============================================================================

struct NonPhotosyntheticCellParam {
    CellState::Type cellType;
    const char* typeName;
};

class NonPhotosyntheticCellTest : public PhotosynthesisTest,
                                   public ::testing::WithParamInterface<NonPhotosyntheticCellParam> {};

TEST_P(NonPhotosyntheticCellTest, NonPlantCellsDoNotPhotosynthesize) {
    const auto param = GetParam();
    OffsetCoord coord{2, 2};
    
    helper.setCellType(coord, param.cellType);
    helper.setLight(coord, 1.0f);
    helper.setPlantWater(coord, 1.0f);
    
    applyPhotosynthesis();
    
    EXPECT_FLOAT_EQ(helper.getPlantSugar(coord), 0.0f)
        << param.typeName << " cells should not photosynthesize";
}

INSTANTIATE_TEST_SUITE_P(
    CellTypes,
    NonPhotosyntheticCellTest,
    ::testing::Values(
        NonPhotosyntheticCellParam{CellState::Type::Air, "Air"},
        NonPhotosyntheticCellParam{CellState::Type::Soil, "Soil"},
        NonPhotosyntheticCellParam{CellState::Type::Dead, "Dead"}
    )
);

TEST_F(PhotosynthesisTest, PlantCellProducesSugarWithLightAndWater) {
    OffsetCoord coord{2, 2};
    
    setupPlantCell(coord, 1.0f, 1.0f);
    
    applyPhotosynthesis();
    
    float expected = expectedSugar(1.0f, 1.0f);
    EXPECT_FLOAT_EQ(helper.getPlantSugar(coord), expected);
    EXPECT_GT(expected, 0.0f);  // Sanity check
}

struct ResourceRequirementParam {
    float light;
    float water;
    const char* description;
};

class ResourceRequirementTest : public PhotosynthesisTest,
                                 public ::testing::WithParamInterface<ResourceRequirementParam> {};

TEST_P(ResourceRequirementTest, NoPhotosynthesisWithoutRequiredResource) {
    const auto param = GetParam();
    OffsetCoord coord{2, 2};
    
    setupPlantCell(coord, param.light, param.water);
    
    applyPhotosynthesis();
    
    EXPECT_FLOAT_EQ(helper.getPlantSugar(coord), 0.0f)
        << "No photosynthesis " << param.description;
}

INSTANTIATE_TEST_SUITE_P(
    MissingResources,
    ResourceRequirementTest,
    ::testing::Values(
        ResourceRequirementParam{0.0f, 1.0f, "without light"},
        ResourceRequirementParam{1.0f, 0.0f, "without water"}
    )
);

// =============================================================================
// Michaelis-Menten Saturation Tests
// =============================================================================

struct SaturationParam {
    float lightMultiplier;  // Multiplier of lightHalfSat
    float waterMultiplier;  // Multiplier of waterHalfSat
    const char* description;
};

class MichaelisMentenTest : public PhotosynthesisTest,
                             public ::testing::WithParamInterface<SaturationParam> {};

TEST_P(MichaelisMentenTest, SaturationFollowsMichaelisMentenKinetics) {
    const auto param = GetParam();
    OffsetCoord coord{2, 2};
    
    float light = param.lightMultiplier * helper.options.lightHalfSat;
    float water = param.waterMultiplier * helper.options.waterHalfSat;
    
    setupPlantCell(coord, light, water);
    applyPhotosynthesis();
    
    float expected = expectedSugar(light, water);
    EXPECT_FLOAT_EQ(helper.getPlantSugar(coord), expected)
        << param.description;
}

INSTANTIATE_TEST_SUITE_P(
    SaturationLevels,
    MichaelisMentenTest,
    ::testing::Values(
        SaturationParam{1.0f, 1.0f, "at half-saturation for both (50% efficiency)"},
        SaturationParam{0.1f, 1.0f, "at low light, high water"},
        SaturationParam{1.0f, 0.1f, "at high light, low water"},
        SaturationParam{10.0f, 1.0f, "at high light saturation"},
        SaturationParam{1.0f, 10.0f, "at high water saturation"}
    )
);

TEST_F(PhotosynthesisTest, HighResourcesApproachMaxRate) {
    OffsetCoord coord{2, 2};
    
    // Very high light and water should approach max rate
    setupPlantCell(coord, 100.0f, 100.0f);
    
    applyPhotosynthesis();
    
    // With very high resources, both terms approach 1.0
    // So production should approach dt * maxRate = 1.0
    float sugar = helper.getPlantSugar(coord);
    EXPECT_NEAR(sugar, helper.options.photoMaxRate * helper.options.dt, 0.02f);
}

// =============================================================================
// Multi-Cell and Accumulation Tests
// =============================================================================

TEST_F(PhotosynthesisTest, MultiplePlantCellsWorkIndependently) {
    // Two plant cells with different light levels
    OffsetCoord coord1{1, 2};
    OffsetCoord coord2{3, 2};
    setupPlantCell(coord1, 1.0f, 1.0f);
    setupPlantCell(coord2, 0.5f, 1.0f);
    
    applyPhotosynthesis();
    
    EXPECT_FLOAT_EQ(helper.getPlantSugar(coord1), expectedSugar(1.0f, 1.0f));
    EXPECT_FLOAT_EQ(helper.getPlantSugar(coord2), expectedSugar(0.5f, 1.0f));
    EXPECT_GT(helper.getPlantSugar(coord1), helper.getPlantSugar(coord2));
}

TEST_F(PhotosynthesisTest, SugarAccumulatesOverMultipleSteps) {
    OffsetCoord coord{2, 2};
    
    setupPlantCell(coord, 1.0f, 1.0f);
    
    // First step
    applyPhotosynthesis();
    float afterFirst = helper.getPlantSugar(coord);
    
    // Second step
    applyPhotosynthesis();
    float afterSecond = helper.getPlantSugar(coord);
    
    EXPECT_FLOAT_EQ(afterSecond, 2.0f * afterFirst);
}

TEST_F(PhotosynthesisTest, TimeStepScalesProduction) {
    OffsetCoord coord{2, 2};
    
    helper.options.dt = 0.5f;
    setupPlantCell(coord, 1.0f, 1.0f);
    
    applyPhotosynthesis();
    
    // With dt=0.5, production should be half
    float expected = 0.5f * helper.options.photoMaxRate 
                   * (1.0f / (1.0f + helper.options.lightHalfSat))
                   * (1.0f / (1.0f + helper.options.waterHalfSat));
    EXPECT_FLOAT_EQ(helper.getPlantSugar(coord), expected);
}

// =============================================================================
// Edge Cases
// =============================================================================

struct LowResourceParam {
    float light;
    float water;
    const char* description;
};

class LowResourceTest : public PhotosynthesisTest,
                         public ::testing::WithParamInterface<LowResourceParam> {};

TEST_P(LowResourceTest, VeryLowResourceProducesMinimalSugar) {
    const auto param = GetParam();
    OffsetCoord coord{2, 2};
    
    setupPlantCell(coord, param.light, param.water);
    
    applyPhotosynthesis();
    
    float sugar = helper.getPlantSugar(coord);
    EXPECT_GT(sugar, 0.0f) << "Should still produce some sugar " << param.description;
    EXPECT_LT(sugar, 0.1f) << "But very little sugar " << param.description;
}

INSTANTIATE_TEST_SUITE_P(
    MinimalProduction,
    LowResourceTest,
    ::testing::Values(
        LowResourceParam{0.01f, 1.0f, "with very low light"},
        LowResourceParam{1.0f, 0.01f, "with very low water"},
        LowResourceParam{0.01f, 0.01f, "with very low light and water"}
    )
);
