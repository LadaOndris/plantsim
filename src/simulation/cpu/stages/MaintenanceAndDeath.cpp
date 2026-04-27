#include "simulation/cpu/stages/MaintenanceAndDeath.h"

MaintenanceAndDeath::MaintenanceAndDeath(const GridShiftHelper& grid)
    : grid(grid)
{
    const int h = grid.height();
    const int w = grid.width();
    
    plantMask.resize(h, w);
    sugarBefore.resize(h, w);
    waterBefore.resize(h, w);
    sugarDeficit.resize(h, w);
    waterDeficit.resize(h, w);
    damage.resize(h, w);
    lightTerm.resize(h, w);
    waterDemand.resize(h, w);
}

void MaintenanceAndDeath::step(State& state, State& backBuffer, const Options& options) {
    if (!options.enableMaintenanceAndDeath) {
        return;
    }

    const int h = grid.height();
    const int w = grid.width();
    const float dt = options.dt;
    const auto& validity = grid.getValidityMask();

    Eigen::Map<const MatrixXi> cellTypes(state.cellTypes.data(), h, w);
    Eigen::Map<const MatrixXf> light(state.light.data(), h, w);
    Eigen::Map<const MatrixXf> sugar(state.plantSugar.data(), h, w);
    Eigen::Map<const MatrixXf> water(state.plantWater.data(), h, w);
    Eigen::Map<const MatrixXf> mineral(state.plantMineral.data(), h, w);
    Eigen::Map<const MatrixXf> health(state.plantHealth.data(), h, w);
    
    Eigen::Map<MatrixXi> nextCellTypes(backBuffer.cellTypes.data(), h, w);
    Eigen::Map<MatrixXf> nextSugar(backBuffer.plantSugar.data(), h, w);
    Eigen::Map<MatrixXf> nextWater(backBuffer.plantWater.data(), h, w);
    Eigen::Map<MatrixXf> nextMineral(backBuffer.plantMineral.data(), h, w);
    Eigen::Map<MatrixXf> nextHealth(backBuffer.plantHealth.data(), h, w);
    Eigen::Map<MatrixXf> nextDeadWater(backBuffer.deadWater.data(), h, w);
    Eigen::Map<MatrixXf> nextDeadMineral(backBuffer.deadMineral.data(), h, w);
    
    nextCellTypes = cellTypes;
    nextSugar = sugar;
    nextWater = water;
    nextMineral = mineral;
    nextHealth = health;
    
    Eigen::Map<const MatrixXf> deadWater(state.deadWater.data(), h, w);
    Eigen::Map<const MatrixXf> deadMineral(state.deadMineral.data(), h, w);
    nextDeadWater = deadWater;
    nextDeadMineral = deadMineral;

    plantMask = (validity.array() * 
                (cellTypes.array() == static_cast<int>(CellState::Type::Cell)).cast<float>()).matrix();

    const float sugarDemand = dt * options.sugarMaintCost;
    
    // Water demand includes light-linked transpiration loss
    lightTerm = (light.array() / (light.array() + options.lightHalfSat + 1e-12f)).matrix();
    waterDemand = (dt * (options.waterMaintCost + options.waterLightLoss * lightTerm.array())).matrix();

    // Store values before deduction for deficit calculation
    sugarBefore = sugar;
    waterBefore = water;

    // Deduct maintenance costs (clamped to 0)
    nextSugar.array() = plantMask.array() * (sugar.array() - sugarDemand).max(0.0f) 
                      + (1.0f - plantMask.array()) * sugar.array();
    
    nextWater.array() = plantMask.array() * (water.array() - waterDemand.array()).max(0.0f)
                      + (1.0f - plantMask.array()) * water.array();

    // Calculate deficits: how much of the demand was unfulfilled
    sugarDeficit = (sugarDemand - sugarBefore.array()).max(0.0f).matrix();
    waterDeficit = (waterDemand.array() - waterBefore.array()).max(0.0f).matrix();

    // Calculate health damage from deficits
    damage = (options.sugarDeficitDamage * sugarDeficit.array() 
            + options.waterDeficitDamage * waterDeficit.array()).matrix();

    // Health regeneration: when no deficit, health regenerates toward 1.0
    // Only regenerate when both sugar and water deficits are zero
    auto noDeficit = (sugarDeficit.array() < 1e-6f) && (waterDeficit.array() < 1e-6f);
    auto healthRegen = noDeficit.cast<float>() * options.healthRegenRate * dt 
                       * (1.0f - health.array()).max(0.0f);

    // Apply damage and regeneration only to plant cells
    nextHealth.array() = plantMask.array() * (health.array() - damage.array() + healthRegen).cwiseMin(1.0f).cwiseMax(0.0f)
                       + (1.0f - plantMask.array()) * health.array();

    // Death conversion: plant cells with health <= 0 become Dead
    // Transfer remaining resources to dead pools
    auto deadMask = (plantMask.array() > 0.5f) && (nextHealth.array() <= 0.0f);
    
    // Convert to Dead cell type
    nextCellTypes = deadMask.select(
        static_cast<int>(CellState::Type::Dead), nextCellTypes);
    
    // Transfer water and mineral to dead pools
    nextDeadWater.array() = deadMask.select(
        nextDeadWater.array() + nextWater.array(), nextDeadWater.array());
    nextDeadMineral.array() = deadMask.select(
        nextDeadMineral.array() + nextMineral.array(), nextDeadMineral.array());
    
    // Clear plant resources for dead cells
    nextSugar.array() = deadMask.select(0.0f, nextSugar.array());
    nextWater.array() = deadMask.select(0.0f, nextWater.array());
    nextMineral.array() = deadMask.select(0.0f, nextMineral.array());
    nextHealth.array() = deadMask.select(0.0f, nextHealth.array());

    // Swap buffers
    std::swap(state.cellTypes, backBuffer.cellTypes);
    std::swap(state.plantSugar, backBuffer.plantSugar);
    std::swap(state.plantWater, backBuffer.plantWater);
    std::swap(state.plantMineral, backBuffer.plantMineral);
    std::swap(state.plantHealth, backBuffer.plantHealth);
    std::swap(state.deadWater, backBuffer.deadWater);
    std::swap(state.deadMineral, backBuffer.deadMineral);
}
