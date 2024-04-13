//
// Created by lada on 4/13/24.
//

#include <algorithm>
#include "WorldStateRenderer.h"


WorldStateRenderer::WorldStateRenderer(const WorldState &worldState, ShaderProgram &program)
        : worldState{worldState},
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
    // TODO: This depends on the specific Map used. Abstract this away.

    auto &map{this->worldState.getMap()};
    auto width = map.getWidth();
    auto height = map.getHeight();
    std::cout << "[Map] Width: " << width << ", height: " << height << std::endl;
    assert (width == height);

    double cellCentersDistance = 1 / static_cast<double>(width - 1);
    double cellRadius = cellCentersDistance / 2;
    double triangleHeight = sqrt(3) / 2 * cellRadius;

    std::vector<GLVertex> vertices;
    std::vector<unsigned int> indices;

    constexpr int indicesPerCell = 18;
    constexpr int verticesPerCell = 7;
    std::vector<unsigned int> singleCellIndices(indicesPerCell); // 6 triangles * 3 vertices = 18
    for (int i = 0; i < 6; ++i) {
        singleCellIndices[i * 3 + 0] = 0; // The center of the cell
        singleCellIndices[i * 3 + 1] = i + 1; // First vertex on the boundary
        singleCellIndices[i * 3 + 2] = (i + 1) % 6 + 1; // Second vertex on the boundary
    }

    float heightFloat = static_cast<float>(height);
    float widthFloat = static_cast<float>(width);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::vector<unsigned int> currentCellIndices{singleCellIndices};
            // Shift the base cell indices to form indices for the current cell.
            std::transform(currentCellIndices.begin(), currentCellIndices.end(), currentCellIndices.begin(),
                           [i, j, height](auto &item) {
                               return item + (i * height + j) * verticesPerCell;
                           });
            indices.insert(indices.end(), currentCellIndices.begin(), currentCellIndices.end());

            float maxY = (heightFloat) * 2 * triangleHeight;
            float maxX = (widthFloat) * 3 / 2 * cellRadius + cellRadius;
            float centerY = static_cast<float>(i) * 2 * triangleHeight / maxY + triangleHeight;
            float centerX = static_cast<float>(j) * 3 / 2 * cellRadius / maxX + cellRadius;

            // Shift the center for specific odd columns
            if (j % 2 == 1) {
                centerY -= triangleHeight;
            }

            vertices.push_back(GLVertex{centerX, centerY, 0});
            float angle = 0.0f;
            for (int k = 0; k < 6; ++k) {
                float x = centerX + cellRadius * cos(k * (2.0f * M_PI) / 6.0f);
                float y = centerY + cellRadius * sin(k * (2.0f * M_PI) / 6.0f);
                vertices.push_back(GLVertex{x, y, 0});
            }
        }
    }

    //std::vector<GLVertex> vertices{GLVertex{0, 1, 0}, GLVertex{0, 0, 0}, GLVertex{1, 0, 0}};
    mapVertexIndices = indices;
    mapVertices = vertices;
}

void WorldStateRenderer::setupVertexArrays() {
    glCreateBuffers(1, &VBO);
    glCreateBuffers(1, &EBO);

    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glNamedBufferData(VBO, mapVertices.size() * sizeof(GLVertex),
                      &mapVertices.front(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glNamedBufferData(EBO, mapVertexIndices.size() * sizeof(unsigned int),
                      &mapVertexIndices.front(), GL_STATIC_DRAW);

    // Set vertex attributes (e.g., position)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *) 0);
    glEnableVertexAttribArray(0);

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
    shaderProgram.use();

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
    glDrawElements(GL_TRIANGLES, mapVertexIndices.size(), GL_UNSIGNED_INT, nullptr);

    auto error = glGetError();
    if (error != GL_NO_ERROR) {
        std::cerr << "[SunRenderer] OpenGL error after rendering: " << error << std::endl;
    }
}
