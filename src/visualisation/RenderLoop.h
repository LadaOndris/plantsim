#pragma once

#include "GraphicsContext.h"
#include "rendering/RendererRegistry.h"
#include "rendering/RenderingOptionsProvider.h"
#include "rendering/GuiFrameRenderer.h"
#include "simulation/ISimulator.h"
#include "ApplicationConfig.h"
#include <functional>

/**
 * @brief Main render loop controller.
 * 
 * Encapsulates the main application loop, coordinating rendering,
 * simulation steps, and event processing.
 */
class RenderLoop {
public:
    /**
     * @brief Function type for creating/resetting the simulation state.
     */
    using StateCreator = std::function<State()>;
    
    /**
     * @brief Run the main application loop.
     * 
     * @param context Graphics context for window management
     * @param registry Renderer registry containing all renderers
     * @param guiRenderer GUI renderer providing simulation control
     * @param simulator Pointer to the current simulator (may be reset)
     * @param config Application configuration
     * @param stateCreator Function to create initial state (for reset)
     */
    static void run(
        GraphicsContext& context,
        RendererRegistry& registry,
        GuiFrameRenderer& guiRenderer,
        std::unique_ptr<ISimulator>& simulator,
        const ApplicationConfig& config,
        StateCreator stateCreator
    );
};
