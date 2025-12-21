#include "simulation/cpu/SoilAbsorption.h"

SoilAbsorption::SoilAbsorption(const GridShiftHelper& grid)
    : grid(grid)
{
    const int h = grid.height();
    const int w = grid.width();
    
    plantMask.resize(h, w);
    uptakeAmount.resize(h, w);
}

void SoilAbsorption::applyAbsorption(
    const MatrixXf& soilResource,
    const MatrixXf& plantResource,
    Eigen::Ref<MatrixXf> nextSoilResource,
    Eigen::Ref<MatrixXf> nextPlantResource,
    float uptakeRate,
    float dt
) {
    // Calculate uptake: plant cells absorb from soil at the same position
    // Desired uptake is uptakeRate * dt, but limited by available soil resource
    MatrixXf desiredUptake = plantMask.array() * uptakeRate * dt;
    uptakeAmount = desiredUptake.cwiseMin(soilResource);
    
    nextSoilResource = soilResource - uptakeAmount;
    nextSoilResource = nextSoilResource.cwiseMax(0.0f);
    
    nextPlantResource = plantResource + uptakeAmount;
}

void SoilAbsorption::step(State& state, State& backBuffer, const Options& options) {
    if (!options.enableSoilSystem) {
        return;
    }

    const int h = grid.height();
    const int w = grid.width();
    const float dt = options.dt;
    const auto& validity = grid.getValidityMask();

    Eigen::Map<const MatrixXi> cellTypes(state.cellTypes.data(), h, w);
    
    plantMask = (validity.array() * 
        (cellTypes.array() == static_cast<int>(CellState::Type::Cell)).cast<float>()).matrix();

    Eigen::Map<const MatrixXf> soilWater(state.soilWater.data(), h, w);
    Eigen::Map<const MatrixXf> soilMineral(state.soilMineral.data(), h, w);
    Eigen::Map<const MatrixXf> plantWater(state.plantWater.data(), h, w);
    Eigen::Map<const MatrixXf> plantMineral(state.plantMineral.data(), h, w);
    
    Eigen::Map<MatrixXf> nextSoilWater(backBuffer.soilWater.data(), h, w);
    Eigen::Map<MatrixXf> nextSoilMineral(backBuffer.soilMineral.data(), h, w);
    Eigen::Map<MatrixXf> nextPlantWater(backBuffer.plantWater.data(), h, w);
    Eigen::Map<MatrixXf> nextPlantMineral(backBuffer.plantMineral.data(), h, w);

    applyAbsorption(soilWater, plantWater, nextSoilWater, nextPlantWater, 
                    options.waterUptakeRate, dt);
    
    applyAbsorption(soilMineral, plantMineral, nextSoilMineral, nextPlantMineral,
                    options.mineralUptakeRate, dt);

    std::swap(state.soilWater, backBuffer.soilWater);
    std::swap(state.soilMineral, backBuffer.soilMineral);
    std::swap(state.plantWater, backBuffer.plantWater);
    std::swap(state.plantMineral, backBuffer.plantMineral);
}
