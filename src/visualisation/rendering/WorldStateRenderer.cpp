//
// Created by lada on 4/13/24.
//

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
    std::vector<GLVertex> vertices{GLVertex{0, 1, 0}, GLVertex{0, 0, 0}, GLVertex{1, 0, 0}};
    mapVertices = vertices;
}

void WorldStateRenderer::setupVertexArrays() {
    glCreateBuffers(1, &VBO);

    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glNamedBufferData(VBO, mapVertices.size() * sizeof(GLVertex),
                      &mapVertices.front(), GL_STATIC_DRAW);

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
}

void WorldStateRenderer::render(const WindowDefinition &window, const RenderingOptions &options) {
    auto &map{this->worldState.getMap()};
    auto width = map.getWidth();
    auto height = map.getHeight();
    std::cout << "[Map] Width: " << width << ", height: " << height << std::endl;

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
    glDrawArrays(GL_TRIANGLES, 0, mapVertices.size());

    auto error = glGetError();
    if (error != GL_NO_ERROR) {
        std::cerr << "[SunRenderer] OpenGL error after rendering: " << error << std::endl;
    }
}
