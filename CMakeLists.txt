cmake_minimum_required(VERSION 3.27)
project(plantsim LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add custom cmake modules path for FindImgui.cmake etc.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# =============================================================================
# Profiling support
# =============================================================================
option(ENABLE_PROFILING "Enable profiling flags for gprof/perf" OFF)

if(ENABLE_PROFILING)
    message(STATUS "Profiling enabled - adding profiling flags")
    # -pg: Generate gprof profiling information
    # -g: Include debug symbols for meaningful profiling output
    # -fno-omit-frame-pointer: Preserve frame pointers for accurate call stacks
    set(PROFILING_FLAGS "-pg -g -fno-omit-frame-pointer")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PROFILING_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${PROFILING_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
endif()

# Output directory for all binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(BUILD_MAIN "Build main application" ON)
option(BUILD_BENCH "Build benchmarks" OFF)
option(BUILD_TEST "Build tests" OFF)

# =============================================================================
# Backend configuration
# Determines which simulator implementations are available (CPU, CUDA, SYCL)
# =============================================================================
set(TARGET_BACKENDS "CPU" CACHE STRING "Comma-separated list of backends: CPU, SYCL, CUDA")
set(VALID_BACKENDS "CPU;SYCL;CUDA")

# Parse and validate backends
string(REPLACE "," ";" BACKEND_LIST "${TARGET_BACKENDS}")

foreach(BACKEND ${BACKEND_LIST})
    string(TOUPPER "${BACKEND}" BACKEND_UPPER)
    list(FIND VALID_BACKENDS "${BACKEND_UPPER}" BACKEND_INDEX)
    if(BACKEND_INDEX EQUAL -1)
        message(FATAL_ERROR "Invalid backend '${BACKEND}'. Valid backends are: ${VALID_BACKENDS}")
    endif()
endforeach()

message(STATUS "Enabled backends: ${BACKEND_LIST}")

# Enable CUDA language if needed (must be done before add_subdirectory)
list(FIND BACKEND_LIST "CUDA" _cuda_index)
if(NOT _cuda_index EQUAL -1)
    if(NOT CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES "61;86" CACHE STRING "CUDA architectures" FORCE)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    message(STATUS "CUDA enabled with architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# =============================================================================
# Subdirectories
# =============================================================================
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
