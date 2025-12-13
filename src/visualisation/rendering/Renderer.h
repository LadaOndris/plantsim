#pragma once

#include "../WindowDefinition.h"
#include "RenderingOptions.h"

class Renderer {
public:
    virtual bool initialize() = 0;

    virtual void render(const WindowDefinition &window, const RenderingOptions &options) = 0;

    virtual void destroy() = 0;

    virtual ~Renderer() = default;
};
