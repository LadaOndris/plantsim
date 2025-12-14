#pragma once

#include "../include/glad/glad.h"
#include <GLFW/glfw3.h>

namespace OpenGLDebug {

/**
 * @brief Get string representation of OpenGL debug source.
 * @see https://github.com/yuzu-emu/yuzu/blob/875568bb3e34725578f7fa3661c8bad89f23a173/src/video_core/renderer_opengl/renderer_opengl.cpp#L82
 */
const char* getSource(GLenum source);

/**
 * @brief Get string representation of OpenGL debug type.
 * @see https://github.com/yuzu-emu/yuzu/blob/875568bb3e34725578f7fa3661c8bad89f23a173/src/video_core/renderer_opengl/renderer_opengl.cpp#L102
 */
const char* getType(GLenum type);

/**
 * @brief OpenGL debug message callback.
 * 
 * Use with glDebugMessageCallback to receive OpenGL debug output.
 */
void APIENTRY debugMessageCallback(
    GLenum source,
    GLenum type,
    GLuint id,
    GLenum severity,
    GLsizei length,
    const GLchar* message,
    const void* userParam
);

/**
 * @brief Enable OpenGL debug output if available.
 */
void enableDebugOutput();

} // namespace OpenGLDebug
