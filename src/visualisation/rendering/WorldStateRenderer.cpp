//
// Created by lada on 4/13/24.
//

#include <algorithm>
#include "WorldStateRenderer.h"
#include "visualisation/rendering/converters/AxialRectangularMapToMeshConverter.h"
#include <GLFW/glfw3.h>

WorldStateRenderer::WorldStateRenderer(const WorldState &worldState, const MapConverter &mapConverter,
                                       ShaderProgram &program)
        : worldState{worldState},
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
    auto &map{this->worldState.getMap()};
    meshData = mapConverter.convert(map);
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
 * Updates the visualization based on the current world state.
 */
void WorldStateRenderer::updateVisualizationInternalState() {
    auto &map{this->worldState.getMap()};

    for (auto &point: map.getPoints()) {
        auto coords = point->getCoords();

        glm::vec3 pointColor = convertPointToColour(*point);

        auto verticesIndices = meshData.cellVerticesMap[coords];

        for (auto &index: verticesIndices) {
            auto colorVector = this->meshData.vertices[index].color;
            colorVector[0] = pointColor[0];
            colorVector[1] = pointColor[1];
            colorVector[2] = pointColor[2];
        }
    }

    auto getRotMatrix = [](float angle) {
        return glm::mat3(glm::rotate(glm::mat4(1.0f), angle, glm::vec3(0.0f, 0.0f, 1.0f)));
    };

    float angle = static_cast<float>(glfwGetTime()) * 2.0f;
    glm::vec3 pointToRotate = glm::vec3(50.f, 5.f, 1.0f);
    glm::vec3 centerOfRotation = glm::vec3(50, 50, 0.0f);
    glm::vec3 rotatedPoint = getRotMatrix(angle) * (pointToRotate - centerOfRotation) + centerOfRotation;
    int pointX = static_cast<int>(rotatedPoint[0]);
    int pointY = static_cast<int>(rotatedPoint[1]);
    std::cout << pointX << " " << pointY << std::endl;

    auto point = map.getPoint(pointX, pointY);
    auto verticesIndices = meshData.cellVerticesMap[point->getCoords()];
    for (auto &index: verticesIndices) {
        auto colorVector = this->meshData.vertices[index].color;
        colorVector[0] = 0.2;
        colorVector[1] = 0.3;
        colorVector[2] = 0.6;
    }

    auto neighbors = map.getNeighbors(point);

    for (auto &neighbor: neighbors) {
        auto verticesIndices = meshData.cellVerticesMap[neighbor->getCoords()];
        for (auto &index: verticesIndices) {
            auto colorVector = this->meshData.vertices[index].color;
            colorVector[0] = 0.2;
            colorVector[1] = 0.6;
            colorVector[2] = 0.3;
        }
    }
}

glm::vec3 WorldStateRenderer::convertPointToColour(const Point &point) const {
    return {0.2, 0.2, 0.2};
}