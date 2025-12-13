
#include <algorithm>
#include "WorldStateRenderer.h"
#include "visualisation/rendering/converters/AxialRectangularMapToMeshConverter.h"
#include <GLFW/glfw3.h>

WorldStateRenderer::WorldStateRenderer(const GridTopology &topology, ISimulator &simulator,
                                       const MapConverter &mapConverter, ShaderProgram &program)
        : topology{topology},
          simulator{simulator},
          mapConverter(mapConverter),
          shaderProgram{program} {
}


bool WorldStateRenderer::initialize() {
    bool isShaderProgramBuilt = shaderProgram.build();
    if (!isShaderProgramBuilt) {
        return false;
    }

    constructVertices();
    setupVertexArrays();

    return true;
}

void WorldStateRenderer::constructVertices() {
    meshData = mapConverter.convert(topology);
}

void WorldStateRenderer::setupVertexArrays() {
    glCreateBuffers(1, &VBO);
    glCreateBuffers(1, &EBO);

    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glNamedBufferData(VBO, meshData.vertices.size() * sizeof(GLVertex),
                      &meshData.vertices.front(), GL_DYNAMIC_DRAW);


    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glNamedBufferData(EBO, meshData.indices.size() * sizeof(unsigned int),
                      &meshData.indices.front(), GL_STATIC_DRAW);

    // Set vertex attributes (e.g., position)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GLVertex), (void *) 0); // position
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(GLVertex), (void *) (3 * sizeof(float))); // color

    glEnableVertexAttribArray(0); // position
    glEnableVertexAttribArray(1); // color

    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        std::cerr << "[SunRenderer] OpenGL error after setting up vertex arrays: " << error << std::endl;
    }
}

void WorldStateRenderer::destroy() {
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
}

void WorldStateRenderer::render(const WindowDefinition &window, const RenderingOptions &options) {
    updateVisualizationInternalState();

    shaderProgram.use();

    glNamedBufferSubData(VBO, 0, meshData.vertices.size() * sizeof(GLVertex),
                         &meshData.vertices.front());

    float left = 0.0f;
    float right = 1.0f;
    float bottom = 0.0f;
    float top = 1.0f;
    float nearVal = -1.0f;
    float farVal = 1.0f;

    glm::mat4 projectionMat = glm::ortho(left, right, bottom, top, nearVal, farVal);

    shaderProgram.setMat4("model", glm::mat4{1.f});
    shaderProgram.setMat4("projection", projectionMat);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDrawElements(GL_TRIANGLES, meshData.indices.size(), GL_UNSIGNED_INT, nullptr);

    auto error = glGetError();
    if (error != GL_NO_ERROR) {
        std::cerr << "[SunRenderer] OpenGL error after rendering: " << error << std::endl;
    }
}

/**
 * Updates the visualization based on the current simulation state.
 */
void WorldStateRenderer::updateVisualizationInternalState() {
    const State &state = simulator.getState();
    StorageCoord storageDims = topology.getStorageDimension();

    for (int r = 0; r < topology.height; r++) {
        for (int q = 0; q < topology.width; q++) {
            OffsetCoord offset{.col = q, .row = r};
            AxialCoord axial = oddrToAxial(offset);
            
            if (!topology.isValid(axial)) {
                continue;
            }

            StorageCoord storageCoord = topology.axialToStorageCoord(axial);
            int idx = storageCoord.asFlat(storageDims);
            
            float pointResources = state.resources[idx];
            auto pointType = static_cast<CellState::Type>(state.cellTypes[idx]);

            glm::vec3 pointColor = convertPointToColour(pointResources, pointType);

            auto verticesIndices = meshData.cellVerticesMap[std::make_pair(r, q)];

            for (auto &index: verticesIndices) {
                auto colorVector = this->meshData.vertices[index].color;
                colorVector[0] = pointColor[0];
                colorVector[1] = pointColor[1];
                colorVector[2] = pointColor[2];
            }
        }
    }
}

glm::vec3 WorldStateRenderer::convertPointToColour(float resources, CellState::Type type) const {
    double resource_factor = fmin(resources / 4.0, 1.0f);
    double G = 0.2f * (type == CellState::Type::Cell);
    double B = 0.2f * (type == CellState::Type::Cell);


    return {resource_factor, G, B};
}