
#include <iostream>
#include "GuiFrameRenderer.h"

GuiFrameRenderer::GuiFrameRenderer() = default;

bool GuiFrameRenderer::initialize() {
    return true;
}

void GuiFrameRenderer::destroy() {
    std::cout << "Destroing" << std::endl;
}

void GuiFrameRenderer::render(const WindowDefinition &window, const RenderingOptions &options) {
    std::cout << "Rendering" << std::endl;
}

RenderingOptions GuiFrameRenderer::getRenderingOptions() const {
    return RenderingOptions{};
}

SimulatorOptions GuiFrameRenderer::getSimulatorOptions() const {
    return SimulatorOptions{};
}
