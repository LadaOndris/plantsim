cmake_minimum_required(VERSION 3.27)
project(plantsim LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

option(BUILD_MAIN "Build main application" ON)
option(BUILD_BENCH "Build benchmarks" OFF)

if(BUILD_MAIN)
    add_subdirectory(src)
    file(COPY shaders/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/shaders)
endif()

if(BUILD_BENCH)
    add_subdirectory(bench)
endif()

if(BUILD_TEST)
    add_subdirectory(test)
endif()
