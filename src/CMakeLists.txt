
set(BINARY ${CMAKE_PROJECT_NAME})

find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(imgui REQUIRED)
find_package(Freetype REQUIRED)
#find_package(Boost 1.57.0 COMPONENTS system filesystem REQUIRED)

include_directories(${GLFW_INCLUDE_DIRS})
include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${FREETYPE_INCLUDE_DIRS})
include_directories(../include)

#if(Boost_FOUND)
#    message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
#    message(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")
#    message(STATUS "Boost_VERSION: ${Boost_VERSION}")
#
#    include_directories(${Boost_INCLUDE_DIRS})
#endif()

file(GLOB_RECURSE SOURCES LIST_DIRECTORIES true *.h *.cpp *.c)
set(SOURCES ${SOURCES})

add_executable(${BINARY}_run ${SOURCES})
target_link_libraries(${BINARY}_run
        ${OPENGL_gl_LIBRARY} glfw glm imgui)

add_library(${BINARY}_lib STATIC ${SOURCES})
target_link_libraries(${BINARY}_lib
        ${OPENGL_gl_LIBRARY} glfw glm imgui)

if(Boost_FOUND)
    target_link_libraries(${BINARY}_run ${Boost_LIBRARIES})
endif()
