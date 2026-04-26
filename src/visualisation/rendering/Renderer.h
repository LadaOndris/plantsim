#pragma once

#include "visualisation/WindowDefinition.h"
#include "RenderingOptions.h"

class Renderer {
public:
    virtual ~Renderer() = default;

    virtual void render(const WindowDefinition &window, const RenderingOptions &options) = 0;
};
