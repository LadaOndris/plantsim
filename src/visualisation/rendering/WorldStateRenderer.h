//
// Created by lada on 4/13/24.
//

#ifndef PLANTSIM_WORLDSTATERENDERER_H
#define PLANTSIM_WORLDSTATERENDERER_H


#include "Renderer.h"
#include "plants/WorldState.h"
#include "GLVertex.h"
#include "visualisation/rendering/shaders/ShaderProgram.h"
#include "visualisation/rendering/converters/MapConverter.h"


class WorldStateRenderer : public Renderer {
public:
    WorldStateRenderer(const WorldState &worldState, const MapConverter &mapConverter, ShaderProgram &program);

    bool initialize() override;

    void destroy() override;

    void render(const WindowDefinition &window, const RenderingOptions &options) override;

private:
    const WorldState &worldState;
    const MapConverter &mapConverter;
    ShaderProgram &shaderProgram;

    unsigned int VAO;
    unsigned int VBO;
    unsigned int EBO;

    std::vector<unsigned int> mapVertexIndices;
    std::vector<GLVertex> mapVertices;

    void constructVertices();

    void setupVertexArrays();
};


#endif //PLANTSIM_WORLDSTATERENDERER_H
