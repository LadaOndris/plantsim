
#pragma once

#include "visualisation/rendering/shaders/ShaderProgram.h"
#include "visualisation/rendering/shaders/Shader.h"

#include <vector>
#include <memory>

class ProgramBuilder {
public:
    ProgramBuilder() = default;

    ProgramBuilder& addShader(const std::string& shaderPath, ShaderType shaderType);

    ShaderProgram build();

private:
    std::vector<std::unique_ptr<Shader>> _shaders;
};