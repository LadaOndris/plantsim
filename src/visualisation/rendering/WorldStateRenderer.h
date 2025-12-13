//
// Created by lada on 4/13/24.
//

#ifndef PLANTSIM_WORLDSTATERENDERER_H
#define PLANTSIM_WORLDSTATERENDERER_H


#include "Renderer.h"
#include "GLVertex.h"
#include "visualisation/rendering/shaders/ShaderProgram.h"
#include "visualisation/rendering/converters/MapConverter.h"
#include "simulation/CellState.h"
#include "simulation/GridTopology.h"
#include "simulation/ISimulator.h"


class WorldStateRenderer : public Renderer {
public:
    WorldStateRenderer(const GridTopology &topology, ISimulator &simulator, 
                       const MapConverter &mapConverter, ShaderProgram &program);

    bool initialize() override;

    void destroy() override;

    void render(const WindowDefinition &window, const RenderingOptions &options) override;

private:
    const GridTopology &topology;
    ISimulator &simulator;
    const MapConverter &mapConverter;
    ShaderProgram &shaderProgram;

    unsigned int VAO;
    unsigned int VBO;
    unsigned int EBO;

    MeshData meshData;

    void constructVertices();

    void setupVertexArrays();

    void updateVisualizationInternalState();

    glm::vec3 convertPointToColour(float resources, CellState::Type type) const;
};


#endif //PLANTSIM_WORLDSTATERENDERER_H
