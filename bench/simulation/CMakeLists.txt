cmake_minimum_required(VERSION 3.27)

set(TARGET_BACKEND "CPU" CACHE STRING "Choose backend: CPU, SYCL, or CUDA")
set_property(CACHE TARGET_BACKEND PROPERTY STRINGS CPU SYCL CUDA)

message(STATUS "Building with backend: ${TARGET_BACKEND}")

# =============================================================================
# Common dependencies
# =============================================================================
find_package(Eigen3 REQUIRED NO_MODULE)

set(BENCH_SIMULATION_BINARY bench_simulation)

file(GLOB_RECURSE SIMULATION_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/simulation/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/plants/*.cpp"
)
# Remove Simulator.cpp: it is obsolete
list(REMOVE_ITEM SIMULATION_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/simulation/Simulator.cpp"
)

# =============================================================================
# Backend-specific configuration
# =============================================================================

if(TARGET_BACKEND STREQUAL "CUDA")
    # CUDA Backend
    # Set CUDA architectures before enabling the language
    # Force set to avoid issues with empty cached values
    if(NOT CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES "61;86" CACHE STRING "CUDA architectures" FORCE)
    endif()
    
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    find_package(CUDAToolkit REQUIRED)
    find_package(OpenMP REQUIRED)
    
    file(GLOB_RECURSE CUDA_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/../../src/simulation/cuda/*.cu"
    )
    
    add_executable(${BENCH_SIMULATION_BINARY} run.cpp ${SIMULATION_SOURCES} ${CUDA_SOURCES})
    
    target_link_libraries(${BENCH_SIMULATION_BINARY} PRIVATE
        CUDA::cudart
        OpenMP::OpenMP_CXX
        Eigen3::Eigen
    )
    
    target_include_directories(${BENCH_SIMULATION_BINARY} PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_compile_definitions(${BENCH_SIMULATION_BINARY} PRIVATE USE_CUDA_BACKEND)
    
    set_target_properties(${BENCH_SIMULATION_BINARY} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

elseif(TARGET_BACKEND STREQUAL "SYCL")

    # SYCL Backend
    find_package(IntelSYCL REQUIRED)
    
    # Manual OpenMP configuration for Intel SYCL compiler
    # Uses Intel's libiomp5 implementation
    set(OpenMP_CXX_FLAGS "-qopenmp")
    set(OpenMP_CXX_LIB_NAMES "libiomp5")
    find_library(OpenMP_libiomp5_LIBRARY
        NAMES iomp5
        HINTS ENV INTEL_ONEAPI_ROOT
        PATH_SUFFIXES compiler/latest/lib
                      compiler/latest/linux/compiler/lib/intel64_lin
                      lib
    )
    
    if(NOT OpenMP_libiomp5_LIBRARY)
        message(WARNING "libiomp5 not found, OpenMP SIMD may not work. Set OpenMP_libiomp5_LIBRARY manually.")
    endif()
    
    add_executable(${BENCH_SIMULATION_BINARY} run.cpp ${SIMULATION_SOURCES})
    
    target_link_libraries(${BENCH_SIMULATION_BINARY} PRIVATE
        sycl
        ${OpenMP_libiomp5_LIBRARY}
        Eigen3::Eigen
    )
    
    target_compile_definitions(${BENCH_SIMULATION_BINARY} PRIVATE USE_SYCL_BACKEND)
    target_compile_options(${BENCH_SIMULATION_BINARY} PRIVATE -fsycl ${OpenMP_CXX_FLAGS})

else()
    # CPU Backend (default)
    find_package(OpenMP REQUIRED)
    
    add_executable(${BENCH_SIMULATION_BINARY} run.cpp ${SIMULATION_SOURCES})
    
    target_link_libraries(${BENCH_SIMULATION_BINARY} PRIVATE
        OpenMP::OpenMP_CXX
        Eigen3::Eigen
    )
    
    target_compile_definitions(${BENCH_SIMULATION_BINARY} PRIVATE USE_CPU_BACKEND)
    
endif()

# =============================================================================
# Common target properties
# =============================================================================
target_include_directories(${BENCH_SIMULATION_BINARY} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    "${EIGEN3_INCLUDE_DIR}"
)

set_target_properties(${BENCH_SIMULATION_BINARY} PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)
