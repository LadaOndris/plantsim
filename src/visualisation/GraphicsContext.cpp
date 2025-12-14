#include "GraphicsContext.h"
#include "OpenGLDebug.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <iostream>

GraphicsContext::~GraphicsContext() {
    if (initialized) {
        shutdown();
    }
}

bool GraphicsContext::initialize(const WindowDefinition& windowDef) {
    windowDefinition = windowDef;

    if (!initializeGlfw(windowDef)) {
        return false;
    }

    if (!initializeGlad()) {
        return false;
    }

    if (!initializeImgui()) {
        return false;
    }

    OpenGLDebug::enableDebugOutput();

    initialized = true;
    return true;
}

void GraphicsContext::shutdown() {
    if (!initialized) {
        return;
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    if (window) {
        glfwDestroyWindow(window);
        window = nullptr;
    }
    glfwTerminate();

    initialized = false;
}

bool GraphicsContext::shouldClose() const {
    return glfwWindowShouldClose(window);
}

void GraphicsContext::swapAndPoll() {
    glfwSwapBuffers(window);
    glfwPollEvents();
}

bool GraphicsContext::initializeGlfw(const WindowDefinition& windowDef) {
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit()) {
        std::cerr << "[ERROR] Couldn't initialize GLFW" << std::endl;
        return false;
    }
    std::cout << "[INFO] GLFW initialized" << std::endl;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifndef NDEBUG
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
#endif
    glfwWindowHint(GLFW_SAMPLES, 4);

    window = glfwCreateWindow(
        windowDef.width, 
        windowDef.height,
        windowDef.name.c_str(), 
        nullptr, 
        nullptr
    );

    if (!window) {
        std::cerr << "[ERROR] Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }

    // Store this pointer for callbacks
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetScrollCallback(window, scrollCallback);

    return true;
}

bool GraphicsContext::initializeGlad() {
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "[ERROR] Failed to initialize GLAD" << std::endl;
        return false;
    }
    std::cout << "[INFO] GLAD initialized" << std::endl;

    return true;
}

bool GraphicsContext::initializeImgui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    (void)io;

    if (!ImGui_ImplGlfw_InitForOpenGL(window, true)) {
        std::cerr << "[ERROR] Failed to initialize ImGui (ImGui_ImplGlfw_InitForOpenGL)" << std::endl;
        return false;
    }

    if (!ImGui_ImplOpenGL3_Init()) {
        std::cerr << "[ERROR] Failed to initialize ImGui (ImGui_ImplOpenGL3_Init)" << std::endl;
        return false;
    }

    std::cout << "[INFO] ImGui initialized" << std::endl;
    return true;
}

void GraphicsContext::errorCallback(int error, const char* description) {
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

void GraphicsContext::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    auto* context = static_cast<GraphicsContext*>(glfwGetWindowUserPointer(window));
    if (context) {
        std::cout << "Window resized to " << width << "/" << height << std::endl;
        glViewport(0, 0, width, height);
        context->windowDefinition.width = width;
        context->windowDefinition.height = height;
    }
}

void GraphicsContext::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    auto& io = ImGui::GetIO();
    if (io.WantCaptureMouse) {
        return;
    }
    // Additional scroll handling can be added here
}
