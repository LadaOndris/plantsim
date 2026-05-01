cmake_minimum_required(VERSION 3.27)
project(plantsim LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(ENABLE_PROFILING "Enable profiling flags for gprof/perf" OFF)
option(BUILD_MAIN "Build main application" ON)
option(BUILD_BENCH "Build benchmarks" OFF)
option(BUILD_TEST "Build tests" OFF)
option(BUILD_EXPERIMENTS "Build experiments" OFF)

set(BACKEND_LIST "CPU" CACHE STRING "Semicolon-separated list of backends to build: CPU, SYCL, CUDA")

if(ENABLE_PROFILING)
    message(STATUS "Profiling enabled")
    add_compile_options(-pg -g -fno-omit-frame-pointer)
    add_link_options(-pg)
endif()

set(VALID_BACKENDS "CPU;SYCL;CUDA")
foreach(BACKEND ${BACKEND_LIST})
    string(TOUPPER "${BACKEND}" BACKEND_UPPER)
    list(FIND VALID_BACKENDS "${BACKEND_UPPER}" _idx)
    if(_idx EQUAL -1)
        message(FATAL_ERROR "Invalid backend '${BACKEND}'. Valid backends: ${VALID_BACKENDS}")
    endif()
endforeach()

message(STATUS "Enabled backends: ${BACKEND_LIST}")

if(BUILD_MAIN OR BUILD_BENCH OR BUILD_TEST OR BUILD_EXPERIMENTS)
    add_subdirectory(src)
endif()

if(BUILD_TEST)
    add_subdirectory(test)
endif()
