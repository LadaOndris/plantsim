#include "RenderLoop.h"
#include "../include/glad/glad.h"

void RenderLoop::run(
    GraphicsContext& context,
    RendererRegistry& registry,
    const RenderingOptionsProvider& optionsProvider,
    ISimulator& simulator,
    const ApplicationConfig& config
) {
    const auto& windowDef = context.getWindowDefinition();

    glViewport(0, 0, windowDef.width, windowDef.height);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);

    while (!context.shouldClose()) {
        for (int i = 0; i < config.stepsPerFrame; ++i) {
            simulator.step(config.simulationOptions);
        }

        RenderingOptions options = optionsProvider.getRenderingOptions();

        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        registry.renderAll(context.getWindowDefinition(), options);

        context.swapAndPoll();
    }
}
