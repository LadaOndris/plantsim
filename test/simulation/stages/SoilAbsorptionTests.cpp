#include "simulation/GridTopology.h"
#include "simulation/State.h"
#include "simulation/Options.h"
#include "simulation/CellState.h"
#include "simulation/cpu/utils/GridShiftHelper.h"
#include "simulation/cpu/stages/SoilAbsorption.h"

#include <gtest/gtest.h>
#include <cmath>
#include <functional>

enum class ResourceType {
    Water,
    Mineral
};

struct ResourceAccessor {
    ResourceType type;
    std::string name;
    float defaultUptakeRate;
    
    float getSoilResource(const State& state, int idx) const {
        return type == ResourceType::Water ? state.soilWater[idx] : state.soilMineral[idx];
    }
    
    float getPlantResource(const State& state, int idx) const {
        return type == ResourceType::Water ? state.plantWater[idx] : state.plantMineral[idx];
    }
    
    float getTotalResource(const State& state) const {
        float total = 0.0f;
        for (size_t i = 0; i < state.totalStorageCells(); ++i) {
            total += getSoilResource(state, i) + getPlantResource(state, i);
        }
        return total;
    }
    
    void setUptakeRate(Options& options, float rate) const {
        if (type == ResourceType::Water) {
            options.waterUptakeRate = rate;
        } else {
            options.mineralUptakeRate = rate;
        }
    }
};

class SoilAbsorptionTest : public ::testing::Test {
protected:
    static constexpr int DEFAULT_GRID_SIZE = 5;
    static constexpr OffsetCoord PLANT_COORD{2, 2};
    static constexpr OffsetCoord DISTANT_COORD{0, 4};
    
    GridTopology topology{DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE};
    
    State createState(const std::vector<OffsetCoord>& plantCoords,
                      const float soilWaterValue = 0.0f,
                      const float soilMineralValue = 0.0f) const
    {
        State s(topology);

        for (const auto& coord : plantCoords) {
            const int storageIdx = topology.toStorageIndex(coord);

            s.cellTypes[storageIdx] = static_cast<int>(CellState::Type::Cell);
            s.soilWater[storageIdx] = soilWaterValue;
            s.soilMineral[storageIdx] = soilMineralValue;
        }

        return s;
    }
    
    State createSinglePlantState(float soilWater = 1.0f, float soilMineral = 0.5f) const {
        return createState({PLANT_COORD}, soilWater, soilMineral);
    }
    
    State createMultiplePlantsState(float soilWater = 1.0f, float soilMineral = 0.5f) const {
        return createState({{1, 2}, {2, 2}, {3, 2}}, soilWater, soilMineral);
    }
    
    Options createOptions(bool enableSoilSystem = true) const {
        Options options;
        options.enableSoilSystem = enableSoilSystem;
        options.waterUptakeRate = 0.08f;
        options.mineralUptakeRate = 0.04f;
        options.dt = 1.0f;
        return options;
    }
    
    void performAbsorptionStep(State& state, State& backBuffer, const Options& options) const {
        GridShiftHelper grid(topology);
        SoilAbsorption absorption(grid);
        absorption.step(state, backBuffer, options);
    }
    
    int getPlantStorageIndex(const OffsetCoord& coord = PLANT_COORD) const {
        return topology.toStorageIndex(coord);
    }
};


class ResourceAbsorptionTest : public SoilAbsorptionTest,
                                public ::testing::WithParamInterface<ResourceAccessor> {};

TEST_P(ResourceAbsorptionTest, PlantAbsorbsResourceFromOverlappingSoil) {
    const auto& resource = GetParam();
    const float initialAmount = 1.0f;
    
    State state = resource.type == ResourceType::Water 
        ? createSinglePlantState(initialAmount, 0.0f)
        : createSinglePlantState(0.0f, initialAmount);
    State backBuffer = state;
    
    const float initialTotal = resource.getTotalResource(state);
    const int plantIdx = getPlantStorageIndex();
    
    performAbsorptionStep(state, backBuffer, createOptions());
    
    // Conservation check
    EXPECT_NEAR(initialTotal, resource.getTotalResource(state), 1e-5f) 
        << resource.name << " should be conserved";
    
    // Plant gained resources
    EXPECT_GT(resource.getPlantResource(state, plantIdx), 0.0f) 
        << "Plant should have absorbed " << resource.name;
    
    // Soil lost resources
    EXPECT_LT(resource.getSoilResource(state, plantIdx), initialAmount) 
        << "Soil should have lost " << resource.name;
}

INSTANTIATE_TEST_SUITE_P(
    BothResources,
    ResourceAbsorptionTest,
    ::testing::Values(
        ResourceAccessor{ResourceType::Water, "water", 0.08f},
        ResourceAccessor{ResourceType::Mineral, "mineral", 0.04f}
    ),
    [](const ::testing::TestParamInfo<ResourceAccessor>& info) {
        return info.param.name;
    }
);

class UptakeRateLimitTest : public SoilAbsorptionTest,
                             public ::testing::WithParamInterface<ResourceAccessor> {};

TEST_P(UptakeRateLimitTest, UptakeRateLimitsAbsorption) {
    const auto& resource = GetParam();
    const float abundantAmount = 10.0f;
    
    State state = resource.type == ResourceType::Water 
        ? createSinglePlantState(abundantAmount, 0.0f)
        : createSinglePlantState(0.0f, abundantAmount);
    State backBuffer = state;
    
    Options options = createOptions();
    resource.setUptakeRate(options, resource.defaultUptakeRate);
    
    performAbsorptionStep(state, backBuffer, options);
    
    const int plantIdx = getPlantStorageIndex();
    const float expectedUptake = resource.defaultUptakeRate * options.dt;
    
    EXPECT_NEAR(resource.getPlantResource(state, plantIdx), expectedUptake, 1e-5f) 
        << resource.name << " uptake should be limited by rate when abundant";
}

INSTANTIATE_TEST_SUITE_P(
    BothResources,
    UptakeRateLimitTest,
    ::testing::Values(
        ResourceAccessor{ResourceType::Water, "water", 0.08f},
        ResourceAccessor{ResourceType::Mineral, "mineral", 0.04f}
    ),
    [](const ::testing::TestParamInfo<ResourceAccessor>& info) {
        return info.param.name;
    }
);

class AvailabilityLimitTest : public SoilAbsorptionTest,
                               public ::testing::WithParamInterface<ResourceAccessor> {};

TEST_P(AvailabilityLimitTest, AbsorptionLimitedByAvailableResources) {
    const auto& resource = GetParam();
    const float scarceAmount = 0.01f;
    
    State state = resource.type == ResourceType::Water 
        ? createSinglePlantState(scarceAmount, 0.0f)
        : createSinglePlantState(0.0f, scarceAmount);
    State backBuffer = state;
    
    Options options = createOptions();
    resource.setUptakeRate(options, 1.0f); // High rate - wants more than available
    
    performAbsorptionStep(state, backBuffer, options);
    
    const int plantIdx = getPlantStorageIndex();
    
    // Soil shouldn't go negative
    EXPECT_GE(resource.getSoilResource(state, plantIdx), 0.0f) 
        << "Soil " << resource.name << " should not go negative";
    
    // Plant absorbs only what's available
    EXPECT_NEAR(resource.getPlantResource(state, plantIdx), scarceAmount, 1e-5f) 
        << "Plant should absorb only available " << resource.name;
}

INSTANTIATE_TEST_SUITE_P(
    BothResources,
    AvailabilityLimitTest,
    ::testing::Values(
        ResourceAccessor{ResourceType::Water, "water", 0.08f},
        ResourceAccessor{ResourceType::Mineral, "mineral", 0.04f}
    ),
    [](const ::testing::TestParamInfo<ResourceAccessor>& info) {
        return info.param.name;
    }
);

TEST_F(SoilAbsorptionTest, DisabledSoilSystemDoesNothing) {
    State state = createSinglePlantState(1.0f, 0.5f);
    State originalState = state;
    State backBuffer = state;
    
    performAbsorptionStep(state, backBuffer, createOptions(false));
    
    EXPECT_EQ(state.soilWater, originalState.soilWater);
    EXPECT_EQ(state.soilMineral, originalState.soilMineral);
    EXPECT_EQ(state.plantWater, originalState.plantWater);
    EXPECT_EQ(state.plantMineral, originalState.plantMineral);
}

TEST_F(SoilAbsorptionTest, ResourceConservationWithMultipleCells) {
    State state = createMultiplePlantsState(1.0f, 0.5f);
    State backBuffer = state;
    
    ResourceAccessor water{ResourceType::Water, "water", 0.08f};
    ResourceAccessor mineral{ResourceType::Mineral, "mineral", 0.04f};
    
    const float initialWater = water.getTotalResource(state);
    const float initialMineral = mineral.getTotalResource(state);
    
    Options options = createOptions();
    
    for (int i = 0; i < 10; ++i) {
        performAbsorptionStep(state, backBuffer, options);
    }
    
    EXPECT_NEAR(initialWater, water.getTotalResource(state), 1e-4f) 
        << "Water should be conserved over multiple steps";
    EXPECT_NEAR(initialMineral, mineral.getTotalResource(state), 1e-4f) 
        << "Mineral should be conserved over multiple steps";
}

TEST_F(SoilAbsorptionTest, WaterAndMineralAbsorbedSimultaneously) {
    const float initialWater = 1.0f;
    const float initialMineral = 0.5f;
    
    State state = createSinglePlantState(initialWater, initialMineral);
    State backBuffer = state;
    
    performAbsorptionStep(state, backBuffer, createOptions());
    
    const int plantIdx = getPlantStorageIndex();
    
    // Both resources absorbed
    EXPECT_GT(state.plantWater[plantIdx], 0.0f);
    EXPECT_GT(state.plantMineral[plantIdx], 0.0f);
    
    // Soil lost both resources
    EXPECT_LT(state.soilWater[plantIdx], initialWater);
    EXPECT_LT(state.soilMineral[plantIdx], initialMineral);
}
