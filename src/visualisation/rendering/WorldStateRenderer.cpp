//
// Created by lada on 4/13/24.
//

#include "WorldStateRenderer.h"

bool WorldStateRenderer::initialize() {
    return true;
}

void WorldStateRenderer::destroy() {

}

void WorldStateRenderer::render(const WindowDefinition &window, const RenderingOptions &options) {
    auto &map{this->worldState.getMap()};
    auto width = map.getWidth();
    auto height = map.getHeight();
    std::cout << "[Map] Width: " << width << ", height: " << height << std::endl;
}

WorldStateRenderer::WorldStateRenderer(const WorldState &worldState) : worldState{worldState} {
}
