#include <iostream>
#include <algorithm>
#include <vector>
#include <memory>

#include "../include/glad/glad.h"
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_stdlib.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "simulation/initializers/regions/CircleRegion.h"
#include "simulation/initializers/regions/GridRegion.h"
#include "visualisation/WindowDefinition.h"
#include "visualisation/rendering/Renderer.h"
#include "visualisation/rendering/RenderingOptionsProvider.h"
#include "visualisation/rendering/GuiFrameRenderer.h"
#include "visualisation/rendering/WorldStateRenderer.h"
#include "visualisation/rendering/converters/AxialRectangularMapToMeshConverter.h"

#include "simulation/ISimulator.h"
#include "simulation/Options.h"
#include "simulation/GridTopology.h"
#include "simulation/State.h"
#include "simulation/initializers/Initializers.h"

#if defined(BACKEND_CPU)
    #include "simulation/cpu/CpuSimulator.h"
#elif defined(BACKEND_CUDA)
    #include "simulation/cuda/CudaSimulator.h"
#elif defined(BACKEND_SYCL)
    #include "simulation/sycl/SyclSimulator.h"
#else
    #error "No backend defined. Define BACKEND_CPU, BACKEND_CUDA, or BACKEND_SYCL"
#endif

namespace {
    GLFWwindow *window{};
    WindowDefinition windowDefinition{};

    void error_callback(int error, const char *description) {
        fprintf(stderr, "Error: %s\n", description);
    }

    void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
        std::cout << "Window resized to " + std::to_string(width) + "/" + std::to_string(height) << std::endl;
        glViewport(0, 0, width, height);
        windowDefinition.width = width;
        windowDefinition.height = height;
    }


    void scroll_callback(GLFWwindow *window, double xoffset, double yoffset) {
        auto &io = ImGui::GetIO();
        if (io.WantCaptureMouse) {
            return;
        }
    }

    /**
     * Creates the window, callbacks, etc.
     */
    bool initializeGlfw() {
        glfwSetErrorCallback(error_callback);

        if (!glfwInit()) {
            std::cerr << "[ERROR] Couldn't initialize GLFW" << std::endl;
            return false;
        } else {
            std::cout << "[INFO] GLFW initialized" << std::endl;
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifndef NDEBUG
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
#endif
        // Multi-sampling
        glfwWindowHint(GLFW_SAMPLES, 4);

        window = glfwCreateWindow(windowDefinition.width, windowDefinition.height,
                                  windowDefinition.name.c_str(), nullptr, nullptr);
        if (!window) {
            std::cout << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            exit(EXIT_FAILURE);
        }

        glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
        glfwSetScrollCallback(window, scroll_callback);
        return true;
    }

    bool initializeGlad() {
        glfwMakeContextCurrent(window);
        if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
            std::cout << "[ERROR] Failed to initialize GLAD" << std::endl;
            return false;
        } else {
            std::cout << "[INFO] GLAD initialized" << std::endl;
        }
        return true;
    }

    bool initializeImgui() {
        std::string fontName = "JetBrainsMono-ExtraLight.ttf";

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();

        ImGuiIO &io = ImGui::GetIO();
        (void) io;

        if (!ImGui_ImplGlfw_InitForOpenGL(window, true)) {
            std::cout << "[ERROR] Failed to initialize ImGui (ImGui_ImplGlfw_InitForOpenGL)" << std::endl;
            return false;
        }
        if (!ImGui_ImplOpenGL3_Init()) {
            std::cout << "[ERROR] Failed to initialize ImGui (ImGui_ImplOpenGL3_Init)" << std::endl;
            return false;
        }

        std::cout << "[INFO] IMGUI initialized" << std::endl;
        return true;
    }

    bool initializeRenderers(std::vector<std::shared_ptr<Renderer>> &renderers) {
        for (const auto &renderer: renderers) {
            bool initializationResult = renderer->initialize();
            if (!initializationResult) {
                auto &rendererObject = *renderer;
                std::cerr << "[ERROR] Failed to initialize a renderer: " << typeid(rendererObject).name() << std::endl;
                return false;
            }
        }
        return true;
    }

    void startRendering(const std::vector<std::shared_ptr<Renderer>> &renderers,
                        const RenderingOptionsProvider &renderingOptionsProvider,
                        ISimulator &simulator) {
        glViewport(0, 0, windowDefinition.width, windowDefinition.height);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_MULTISAMPLE);

        Options simOptions{
            .enableResourceTransfer = true,
            .enableCellMultiplication = true
        };

        while (!glfwWindowShouldClose(window)) {
            for (int i = 0; i < 100; i++) {
                simulator.step(simOptions);
            }

            RenderingOptions options{renderingOptionsProvider.getRenderingOptions()};

            glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            for (const auto &renderer: renderers) {
                renderer->render(windowDefinition, options);
            }

            glfwSwapBuffers(window);
            glfwPollEvents();
        }
    }

    void destroyRenderers(const std::vector<std::shared_ptr<Renderer>> &renderers) {
        for (const auto &renderer: renderers) {
            renderer->destroy();
        }
    }

    void cleanup() {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        glfwDestroyWindow(window);
        glfwTerminate();
    }


    /**
     * From:
     * https://github.com/yuzu-emu/yuzu/blob/875568bb3e34725578f7fa3661c8bad89f23a173/src/video_core/renderer_opengl/renderer_opengl.cpp#L82
     */
    const char *getSource(GLenum source) {
        switch (source) {
            case GL_DEBUG_SOURCE_API:
                return "API";
            case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
                return "WINDOW_SYSTEM";
            case GL_DEBUG_SOURCE_SHADER_COMPILER:
                return "SHADER_COMPILER";
            case GL_DEBUG_SOURCE_THIRD_PARTY:
                return "THIRD_PARTY";
            case GL_DEBUG_SOURCE_APPLICATION:
                return "APPLICATION";
            case GL_DEBUG_SOURCE_OTHER:
                return "OTHER";
            default:
                return "Unknown source";
        }
    }

    /**
     * From:
     * https://github.com/yuzu-emu/yuzu/blob/875568bb3e34725578f7fa3661c8bad89f23a173/src/video_core/renderer_opengl/renderer_opengl.cpp#L102
     */
    const char *getType(GLenum type) {
        switch (type) {
            case GL_DEBUG_TYPE_ERROR:
                return "ERROR";
            case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
                return "DEPRECATED_BEHAVIOR";
            case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
                return "UNDEFINED_BEHAVIOR";
            case GL_DEBUG_TYPE_PORTABILITY:
                return "PORTABILITY";
            case GL_DEBUG_TYPE_PERFORMANCE:
                return "PERFORMANCE";
            case GL_DEBUG_TYPE_OTHER:
                return "OTHER";
            case GL_DEBUG_TYPE_MARKER:
                return "MARKER";
            default:
                return "Unknown type";
        }
    }

    /**
     * From:
     * https://github.com/yuzu-emu/yuzu/blob/875568bb3e34725578f7fa3661c8bad89f23a173/src/video_core/renderer_opengl/renderer_opengl.cpp#L102
     */
    void APIENTRY processErrorMessageCallback(
            GLenum source,
            GLenum type,
            GLuint id,
            GLenum severity,
            GLsizei length,
            const GLchar *message,
            const void *userParam
    ) {
        const char format[] = "%s %s %u: %s";
        const char *const str_source = getSource(source);
        const char *const str_type = getType(type);

        fprintf(stderr, format, str_source, str_type, id, message);
    }

    std::unique_ptr<ISimulator> createSimulator(State initialState) {
#if defined(BACKEND_CPU)
std::cout << "[INFO] Using CPU backend" << std::endl;
        return std::make_unique<CpuSimulator>(std::move(initialState));
#elif defined(BACKEND_CUDA)
std::cout << "[INFO] Using CUDA backend" << std::endl;
        return std::make_unique<CudaSimulator>(std::move(initialState));
#elif defined(BACKEND_SYCL)
std::cout << "[INFO] Using SYCL backend" << std::endl;
        return std::make_unique<SyclSimulator>(std::move(initialState));
#endif
    }

    int runApplication() {
        using namespace initializers;

        constexpr int gridSize = 200;
        GridTopology topology{gridSize, gridSize};

        OffsetCoord center{gridSize / 2, gridSize / 2};
        StateInitializer initializer{
            // Set all cells to Cell type
            //PolicyApplication{CircleRegion{center, 70}, SetCellType{CellState::Cell}},
            PolicyApplication{GridRegion{10, 70, center.col - 5, center.row - 35}, SetCellType{CellState::Cell}},
            // Set resources at center cell
            PolicyApplication{
                SingleCell{center},
                SetResource{FixedAmount{10000.0f}}
            }
        };
        State initialState = initializer.initialize(topology);

        std::unique_ptr<ISimulator> simulator = createSimulator(std::move(initialState));

        std::vector<std::shared_ptr<Renderer>> renderers;

        ShaderProgram worldStateRendererProgram;
        worldStateRendererProgram.addShader(
                std::make_unique<Shader>("./shaders/map/shader.vert", ShaderType::Vertex)
        );
        worldStateRendererProgram.addShader(
                std::make_unique<Shader>("./shaders/map/shader.frag", ShaderType::Fragment)
        );

        AxialRectangularMapToMeshConverter mapConverter{};
        auto worldStateRenderer{
                std::make_shared<WorldStateRenderer>(topology, *simulator, mapConverter, worldStateRendererProgram)};
        renderers.push_back(worldStateRenderer);

        auto guiFrameRenderer{std::make_shared<GuiFrameRenderer>()};
        renderers.push_back(guiFrameRenderer);

        bool result = initializeRenderers(renderers);
        if (!result) {
            destroyRenderers(renderers);
            return EXIT_FAILURE;
        }

        startRendering(renderers, *guiFrameRenderer, *simulator);

        destroyRenderers(renderers);
        return EXIT_SUCCESS;
    }
}

int main() {
    std::cout << "Starting the application..." << std::endl;
    if (!initializeGlfw() || !initializeGlad() || !initializeImgui()) {
        return EXIT_FAILURE;
    }

    if (glDebugMessageCallback) {
        glEnable(GL_DEBUG_OUTPUT);
        glDebugMessageCallback(processErrorMessageCallback, nullptr);
        std::cout << "[INFO] OpenGL debug output enabled" << std::endl;
    }

    int returnCode = runApplication();

    cleanup();
    return returnCode;
}