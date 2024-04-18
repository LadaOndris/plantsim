
#include <iostream>
#include "GuiFrameRenderer.h"

GuiFrameRenderer::GuiFrameRenderer() = default;

bool GuiFrameRenderer::initialize() {
    return true;
}

void GuiFrameRenderer::destroy() {
}

void GuiFrameRenderer::render(const WindowDefinition &window, const RenderingOptions &options) {
}

RenderingOptions GuiFrameRenderer::getRenderingOptions() const {
    return RenderingOptions{};
}

SimulatorOptions GuiFrameRenderer::getSimulatorOptions() const {
    return SimulatorOptions{};
}
