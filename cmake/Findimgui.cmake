# FindImgui.cmake
# Finds the imgui library and creates a target for it.
#
# This module defines:
#   imgui_FOUND       - True if imgui was found
#   imgui_INCLUDE_DIR - The imgui include directory
#   imgui             - The imgui library target
#
# Usage:
#   Set IMGUI_DIR to point to the imgui source directory, or set it via
#   environment variable, or let it default to ~/tmp/imgui

# Try to find imgui directory
find_path(imgui_INCLUDE_DIR
    NAMES imgui.h
    PATHS
        ${IMGUI_DIR}
        $ENV{IMGUI_DIR}
        $ENV{HOME}/tmp/imgui
        /usr/local/include
        /usr/include
    DOC "Path to imgui include directory"
)

if(imgui_INCLUDE_DIR)
    set(imgui_FOUND TRUE)
    
    # Collect imgui source files
    set(IMGUI_SOURCES
        ${imgui_INCLUDE_DIR}/imgui.cpp
        ${imgui_INCLUDE_DIR}/imgui_demo.cpp
        ${imgui_INCLUDE_DIR}/imgui_draw.cpp
        ${imgui_INCLUDE_DIR}/imgui_tables.cpp
        ${imgui_INCLUDE_DIR}/imgui_widgets.cpp
    )
    
    # Add backend implementations for GLFW and OpenGL3
    set(IMGUI_BACKEND_SOURCES
        ${imgui_INCLUDE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_INCLUDE_DIR}/backends/imgui_impl_opengl3.cpp
    )
    
    # Check if all source files exist
    foreach(src ${IMGUI_SOURCES} ${IMGUI_BACKEND_SOURCES})
        if(NOT EXISTS ${src})
            message(WARNING "imgui source file not found: ${src}")
            set(imgui_FOUND FALSE)
        endif()
    endforeach()
    
    if(imgui_FOUND)
        # Create the imgui library target if it doesn't exist
        if(NOT TARGET imgui)
            add_library(imgui STATIC
                ${IMGUI_SOURCES}
                ${IMGUI_BACKEND_SOURCES}
            )
            
            target_include_directories(imgui PUBLIC
                ${imgui_INCLUDE_DIR}
                ${imgui_INCLUDE_DIR}/backends
            )
            
            # imgui backends need GLFW and OpenGL
            find_package(glfw3 QUIET)
            find_package(OpenGL QUIET)
            
            if(TARGET glfw)
                target_link_libraries(imgui PUBLIC glfw)
            endif()
            
            if(OpenGL_FOUND)
                target_link_libraries(imgui PUBLIC OpenGL::GL)
            endif()
            
            # Set C++ standard
            set_target_properties(imgui PROPERTIES
                CXX_STANDARD 11
                CXX_STANDARD_REQUIRED ON
            )
        endif()
        
        message(STATUS "Found imgui: ${imgui_INCLUDE_DIR}")
    endif()
else()
    set(imgui_FOUND FALSE)
endif()

# Handle REQUIRED and QUIET arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(imgui
    REQUIRED_VARS imgui_INCLUDE_DIR imgui_FOUND
)

mark_as_advanced(imgui_INCLUDE_DIR)
