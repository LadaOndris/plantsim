#pragma once

#include "visualisation/WindowDefinition.h"
#include "simulation/Options.h"
#include <string>

/**
 * @brief Application-wide configuration settings.
 * 
 * Contains all configurable parameters for the application,
 * including window settings, simulation parameters, and rendering options.
 */
struct ApplicationConfig {
    WindowDefinition window{};

    int gridWidth{200};
    int gridHeight{200};

    Options simulationOptions{
        .enableResourceTransfer = true,
        .enableCellMultiplication = true,
        .enableSoilSystem = true,
        .enableMaintenanceAndDeath = true,
        .enableDeadDecay = true
    };

    int stepsPerFrame{50};

    std::string vertexShaderPath{"shaders/map/shader.vert"};
    std::string fragmentShaderPath{"shaders/map/shader.frag"};
};
