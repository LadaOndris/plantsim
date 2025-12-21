
#include <algorithm>
#include "WorldStateRenderer.h"
#include "visualisation/rendering/converters/AxialRectangularMapToMeshConverter.h"
#include <GLFW/glfw3.h>

WorldStateRenderer::WorldStateRenderer(const GridTopology &topology, 
                                       std::unique_ptr<ISimulator>& simulatorPtr,
                                       const MapConverter &mapConverter, 
                                       std::shared_ptr<ShaderProgram> program)
        : topology{topology},
          simulatorPtr{simulatorPtr},
          mapConverter(mapConverter),
          shaderProgram{std::move(program)} {
}


bool WorldStateRenderer::initialize() {
    bool isShaderProgramBuilt = shaderProgram->build();
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
    updateVisualizationInternalState(options);

    shaderProgram->use();

    glNamedBufferSubData(VBO, 0, meshData.vertices.size() * sizeof(GLVertex),
                         &meshData.vertices.front());

    float left = 0.0f;
    float right = 1.0f;
    float bottom = 0.0f;
    float top = 1.0f;
    float nearVal = -1.0f;
    float farVal = 1.0f;

    glm::mat4 projectionMat = glm::ortho(left, right, bottom, top, nearVal, farVal);

    shaderProgram->setMat4("model", glm::mat4{1.f});
    shaderProgram->setMat4("projection", projectionMat);

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
void WorldStateRenderer::updateVisualizationInternalState(const RenderingOptions& options) {
    const State &state = simulatorPtr->getState();
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
            
            float plantSugar = state.plantSugar[idx];
            float pointSoilWater = state.soilWater[idx];
            float pointSoilMineral = state.soilMineral[idx];
            auto pointType = static_cast<CellState::Type>(state.cellTypes[idx]);

            glm::vec3 pointColor = computeCellColor(plantSugar, pointSoilWater, pointSoilMineral, pointType, options);

            auto &verticesIndices = meshData.cellVerticesMap[std::make_pair(r, q)];

            for (auto &index: verticesIndices) {
                auto& colorVector = this->meshData.vertices[index].color;
                colorVector[0] = pointColor[0];
                colorVector[1] = pointColor[1];
                colorVector[2] = pointColor[2];
            }
        }
    }
}

glm::vec3 WorldStateRenderer::computeCellColor(float resources, float soilWater, float soilMineral,
                                                CellState::Type type, 
                                                const RenderingOptions& options) const {
    glm::vec3 color{0.0f};
    float totalOpacity = 0.0f;

    constexpr glm::vec3 CELL_COLOR{0.1f, 0.6f, 0.2f};
    constexpr glm::vec3 AIR_COLOR{0.05f, 0.05f, 0.05f};
    constexpr glm::vec3 RESOURCE_BASE_COLOR{1.0f, 0.0f, 0.0f};  // Red
    constexpr glm::vec3 WATER_BASE_COLOR{0.0f, 0.4f, 1.0f};     // Blue
    constexpr glm::vec3 MINERAL_BASE_COLOR{0.6f, 0.3f, 0.1f};   // Brown/Orange
    constexpr float RESOURCE_MAX = 1.0f;
    constexpr float WATER_MAX = 2.0f;
    constexpr float MINERAL_MAX = 2.0f;

    auto blendLayer = [&](bool enabled, const glm::vec3& layerColor, float layerOpacity) {
        if (enabled && layerOpacity > 0.0f) {
            color += layerColor * layerOpacity;
            totalOpacity += layerOpacity;
        }
    };

    // Layer 1: Cell Types
    if (options.showCellTypes) {
        glm::vec3 cellTypeColor = (type == CellState::Type::Cell) ? CELL_COLOR : AIR_COLOR;
        blendLayer(true, cellTypeColor, options.cellTypesOpacity);
    }

    // Layer 2: Resources (Red gradient)
    if (options.showResources && resources > 0.0f) {
        float intensity = std::min(resources / RESOURCE_MAX, 1.0f);
        glm::vec3 resourceColor = RESOURCE_BASE_COLOR * intensity;
        blendLayer(true, resourceColor, options.resourcesOpacity * intensity);
    }

    // Layer 3: Soil Water (Blue gradient)
    if (options.showSoilWater && soilWater > 0.0f) {
        float intensity = std::min(soilWater / WATER_MAX, 1.0f);
        glm::vec3 waterColor = WATER_BASE_COLOR * intensity;
        blendLayer(true, waterColor, options.soilWaterOpacity * intensity);
    }

    // Layer 4: Soil Mineral (Brown/Orange gradient)
    if (options.showSoilMineral && soilMineral > 0.0f) {
        float intensity = std::min(soilMineral / MINERAL_MAX, 1.0f);
        glm::vec3 mineralColor = MINERAL_BASE_COLOR * intensity;
        blendLayer(true, mineralColor, options.soilMineralOpacity * intensity);
    }

    // Normalize if total opacity exceeds 1
    if (totalOpacity > 1.0f) {
        color /= totalOpacity;
    }

    return glm::clamp(color, 0.0f, 1.0f);
}
