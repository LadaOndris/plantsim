#pragma once

#include "RenderingOptions.h"

class RenderingOptionsProvider {
public:
    virtual RenderingOptions getRenderingOptions() const = 0;
};
