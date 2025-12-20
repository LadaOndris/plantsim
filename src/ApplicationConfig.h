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
    /// Window configuration
    WindowDefinition window{};

    /// Simulation grid dimensions
    int gridWidth{200};
    int gridHeight{200};

    /// Simulation options passed to simulator each step
    Options simulationOptions{
        .enableResourceTransfer = true,
        .enableCellMultiplication = true,
        .enableSoilSystem = true
    };

    /// Number of simulation steps per render frame
    int stepsPerFrame{50};

    /// Shader paths
    std::string vertexShaderPath{"./shaders/map/shader.vert"};
    std::string fragmentShaderPath{"./shaders/map/shader.frag"};
};
