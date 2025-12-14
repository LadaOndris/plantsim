#pragma once

#include "GraphicsContext.h"
#include "rendering/RendererRegistry.h"
#include "rendering/RenderingOptionsProvider.h"
#include "simulation/ISimulator.h"
#include "ApplicationConfig.h"

/**
 * @brief Main render loop controller.
 * 
 * Encapsulates the main application loop, coordinating rendering,
 * simulation steps, and event processing.
 */
class RenderLoop {
public:
    /**
     * @brief Run the main application loop.
     * 
     * @param context Graphics context for window management
     * @param registry Renderer registry containing all renderers
     * @param optionsProvider Provider for rendering options
     * @param simulator The simulation to step
     * @param config Application configuration
     */
    static void run(
        GraphicsContext& context,
        RendererRegistry& registry,
        const RenderingOptionsProvider& optionsProvider,
        ISimulator& simulator,
        const ApplicationConfig& config
    );
};
