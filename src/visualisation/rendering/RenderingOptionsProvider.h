
#ifndef PLANTSIM_RENDERINGOPTIONSPROVIDER_H
#define PLANTSIM_RENDERINGOPTIONSPROVIDER_H


#include "RenderingOptions.h"

class RenderingOptionsProvider {
public:
    virtual RenderingOptions getRenderingOptions() const = 0;
};


#endif
