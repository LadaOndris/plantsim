find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(imgui REQUIRED)
find_package(Freetype REQUIRED)
find_package(glad REQUIRED)

set(IMGUI_BACKEND_SOURCES
    imgui_backends/imgui_impl_glfw.cpp
    imgui_backends/imgui_impl_glfw.h
    imgui_backends/imgui_impl_opengl3.cpp
    imgui_backends/imgui_impl_opengl3.h
)

add_library(plantsim_visualisation STATIC
    GraphicsContext.cpp
    GraphicsContext.h
    OpenGLDebug.cpp
    OpenGLDebug.h
    RenderLoop.cpp
    RenderLoop.h
    WindowDefinition.cpp
    WindowDefinition.h
    rendering/Renderer.cpp
    rendering/Renderer.h
    rendering/RendererRegistry.h
    rendering/RenderingOptions.cpp
    rendering/RenderingOptions.h
    rendering/RenderingOptionsProvider.cpp
    rendering/RenderingOptionsProvider.h
    rendering/GuiFrameRenderer.cpp
    rendering/GuiFrameRenderer.h
    rendering/WorldStateRenderer.cpp
    rendering/WorldStateRenderer.h
    rendering/GLVertex.cpp
    rendering/GLVertex.h
    rendering/SimulationControl.h
    rendering/SliderHelper.h
    rendering/converters/MapConverter.cpp
    rendering/converters/MapConverter.h
    rendering/converters/MeshData.cpp
    rendering/converters/MeshData.h
    rendering/converters/AxialRectangularMapToMeshConverter.cpp
    rendering/converters/AxialRectangularMapToMeshConverter.h
    rendering/converters/OffsetRectangularMapToMeshConverter.cpp
    rendering/converters/OffsetRectangularMapToMeshConverter.h
    rendering/shaders/Shader.cpp
    rendering/shaders/Shader.h
    rendering/shaders/ShaderProgram.cpp
    rendering/shaders/ShaderProgram.h
    rendering/shaders/ProgramBuilder.cpp
    rendering/shaders/ProgramBuilder.h
    ${IMGUI_BACKEND_SOURCES}
)
target_include_directories(plantsim_visualisation PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/imgui_backends)

target_link_libraries(plantsim_visualisation PUBLIC
    OpenGL::GL
    glfw
    glm::glm
    imgui::imgui
    Freetype::Freetype
    glad::glad
    plantsim_common
    plantsim_resources
)
