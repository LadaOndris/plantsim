
#ifndef PLANTSIM_GUIFRAMERENDERER_H
#define PLANTSIM_GUIFRAMERENDERER_H


#include "Renderer.h"
#include "RenderingOptionsProvider.h"
#include "simulation/SimulatorOptionsProvider.h"

class GuiFrameRenderer : public Renderer, public RenderingOptionsProvider, public SimulatorOptionsProvider {
public:
    GuiFrameRenderer();

    bool initialize() override;

    void destroy() override;

    void render(const WindowDefinition &window, const RenderingOptions &options) override;

    RenderingOptions getRenderingOptions() const override;

    SimulatorOptions getSimulatorOptions() const override;
};


#endif
