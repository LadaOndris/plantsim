#pragma once

#include "../include/glad/glad.h"
#include <GLFW/glfw3.h>
#include "WindowDefinition.h"

/**
 * @brief RAII wrapper for graphics context initialization.
 * 
 * Manages the lifecycle of GLFW window, GLAD loader, and ImGui.
 * Encapsulates all low-level graphics initialization and cleanup.
 */
class GraphicsContext {
public:
    GraphicsContext() = default;
    ~GraphicsContext();

    // Non-copyable, non-movable (owns OpenGL context)
    GraphicsContext(const GraphicsContext&) = delete;
    GraphicsContext& operator=(const GraphicsContext&) = delete;
    GraphicsContext(GraphicsContext&&) = delete;
    GraphicsContext& operator=(GraphicsContext&&) = delete;

    /**
     * @brief Initialize all graphics subsystems.
     * 
     * @param windowDef Window configuration
     * @return true if all subsystems initialized successfully
     */
    bool initialize(const WindowDefinition& windowDef);

    /**
     * @brief Shutdown all graphics subsystems.
     * 
     * Called automatically by destructor, but can be called manually.
     */
    void shutdown();

    /**
     * @brief Get the GLFW window handle.
     */
    [[nodiscard]] GLFWwindow* getWindow() const { return window; }

    /**
     * @brief Get current window definition (may change on resize).
     */
    [[nodiscard]] const WindowDefinition& getWindowDefinition() const { return windowDefinition; }

    /**
     * @brief Get mutable window definition reference for resize callbacks.
     */
    [[nodiscard]] WindowDefinition& getWindowDefinition() { return windowDefinition; }

    /**
     * @brief Check if window should close.
     */
    [[nodiscard]] bool shouldClose() const;

    /**
     * @brief Swap buffers and poll events.
     */
    void swapAndPoll();

private:
    bool initializeGlfw(const WindowDefinition& windowDef);
    bool initializeGlad();
    bool initializeImgui();

    static void errorCallback(int error, const char* description);
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);

    GLFWwindow* window{nullptr};
    WindowDefinition windowDefinition{};
    bool initialized{false};
};
