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
    WorldStateRenderer(WorldState &worldState, const MapConverter &mapConverter, ShaderProgram &program);

    bool initialize() override;

    void destroy() override;

    void render(const WindowDefinition &window, const RenderingOptions &options) override;

private:
    WorldState &worldState;
    const MapConverter &mapConverter;
    ShaderProgram &shaderProgram;

    unsigned int VAO;
    unsigned int VBO;
    unsigned int EBO;

    MeshData meshData;

    void constructVertices();

    void setupVertexArrays();

    void updateVisualizationInternalState();

    glm::vec3 convertPointToColour(int resources, Point::Type type) const;
};


#endif //PLANTSIM_WORLDSTATERENDERER_H
