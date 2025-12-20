#pragma once

#include "Renderer.h"
#include "RenderingOptionsProvider.h"
#include "RenderingOptions.h"
#include "SimulationControl.h"
#include "simulation/Options.h"

/**
 * @brief ImGui-based GUI renderer for simulation control.
 */
class GuiFrameRenderer : public Renderer, public RenderingOptionsProvider {
public:
    GuiFrameRenderer();

    void initializeWithOptions(const Options& initialOptions, int stepsPerFrame);

    bool initialize() override;

    void destroy() override;

    void render(const WindowDefinition &window, const RenderingOptions &options) override;

    RenderingOptions getRenderingOptions() const override;
    
    SimulationControl& getSimulationControl();
    
    const SimulationControl& getSimulationControl() const;

private:
    RenderingOptions renderingOptions{};
    SimulationControl simulationControl{};
    
    void renderPlaybackControls();
    void renderVisualizationControls();
    void renderOptionsPanel();
    void renderPendingChangesIndicator();
};
