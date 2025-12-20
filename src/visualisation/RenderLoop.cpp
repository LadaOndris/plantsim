#include "RenderLoop.h"
#include "../include/glad/glad.h"
#include "simulation/SimulatorFactory.h"

void RenderLoop::run(
    GraphicsContext& context,
    RendererRegistry& registry,
    GuiFrameRenderer& guiRenderer,
    std::unique_ptr<ISimulator>& simulator,
    const ApplicationConfig& config,
    StateCreator stateCreator
) {
    const auto& windowDef = context.getWindowDefinition();

    glViewport(0, 0, windowDef.width, windowDef.height);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);

    while (!context.shouldClose()) {
        SimulationControl& control = guiRenderer.getSimulationControl();
        
        // Handle reset request
        if (control.consumeResetFlag()) {
            // Apply pending options
            control.applyPendingOptions();
            
            // Recreate state and simulator
            State newState = stateCreator();
            simulator = SimulatorFactory::create(std::move(newState), control.activeOptions);
        }
        
        // Run simulation steps if not paused
        if (!control.isPaused) {
            for (int i = 0; i < control.stepsPerFrame; ++i) {
                simulator->step(control.activeOptions);
            }
        }

        RenderingOptions options = guiRenderer.getRenderingOptions();

        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        registry.renderAll(context.getWindowDefinition(), options);

        context.swapAndPoll();
    }
}
