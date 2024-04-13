//
// Created by lada on 4/13/24.
//

#ifndef PLANTSIM_WORLDSTATERENDERER_H
#define PLANTSIM_WORLDSTATERENDERER_H


#include "Renderer.h"
#include "plants/WorldState.h"

class WorldStateRenderer : public Renderer {
public:
    WorldStateRenderer(const WorldState &worldState);

    bool initialize() override;

    void destroy() override;

    void render(const WindowDefinition &window, const RenderingOptions &options) override;

private:
    const WorldState &worldState;
};


#endif //PLANTSIM_WORLDSTATERENDERER_H
