#pragma once

#include "Renderer.h"
#include "GLVertex.h"
#include "RenderingOptions.h"
#include "visualisation/rendering/shaders/ShaderProgram.h"
#include "visualisation/rendering/converters/MapConverter.h"
#include "simulation/CellState.h"
#include "simulation/GridTopology.h"
#include "simulation/ISimulator.h"
#include <memory>


class WorldStateRenderer : public Renderer {
public:
    WorldStateRenderer(const GridTopology &topology, 
                       std::unique_ptr<ISimulator>& simulatorPtr, 
                       const MapConverter &mapConverter, 
                       std::shared_ptr<ShaderProgram> program);

    bool initialize() override;

    void destroy() override;

    void render(const WindowDefinition &window, const RenderingOptions &options) override;

private:
    const GridTopology &topology;
    std::unique_ptr<ISimulator>& simulatorPtr;
    const MapConverter &mapConverter;
    std::shared_ptr<ShaderProgram> shaderProgram;

    unsigned int VAO;
    unsigned int VBO;
    unsigned int EBO;

    MeshData meshData;

    void constructVertices();

    void setupVertexArrays();

    void updateVisualizationInternalState(const RenderingOptions& options);

    glm::vec3 computeCellColor(float sugar, float water, float mineral,
                               CellState::Type type, const RenderingOptions& options) const;
};
