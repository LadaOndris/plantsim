
#include "visualisation/rendering/shaders/ProgramBuilder.h"

#include <memory>

ProgramBuilder& ProgramBuilder::addShader(const std::string& shaderPath, ShaderType shaderType) {
    _shaders.push_back(std::make_unique<Shader>(shaderPath, shaderType));
    return *this;
}

ShaderProgram ProgramBuilder::build() {
    return ShaderProgram(std::move(_shaders));
}