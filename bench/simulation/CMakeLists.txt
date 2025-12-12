cmake_minimum_required(VERSION 3.27)

# BACKEND_LIST is set in top-level CMakeLists.txt
message(STATUS "Building simulation benchmarks with backends: ${BACKEND_LIST}")

# =============================================================================
# Source files (defined once, not in function)
# =============================================================================
set(PLANTSIM_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../src")
set(PLANTSIM_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../include")

file(GLOB_RECURSE SIMULATION_SOURCES
    "${PLANTSIM_SRC_DIR}/simulation/*.cpp"
    "${PLANTSIM_SRC_DIR}/plants/*.cpp"
)
list(FILTER SIMULATION_SOURCES EXCLUDE REGEX ".*Simulator\.cpp$")

file(GLOB_RECURSE CUDA_SOURCES "${PLANTSIM_SRC_DIR}/simulation/cuda/*.cu")

# =============================================================================
# Find packages once (not inside function)
# =============================================================================
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)

list(FIND BACKEND_LIST "CUDA" _has_cuda)
if(NOT _has_cuda EQUAL -1)
    find_package(CUDAToolkit REQUIRED)
endif()

list(FIND BACKEND_LIST "SYCL" _has_sycl)
if(NOT _has_sycl EQUAL -1)
    find_package(IntelSYCL REQUIRED)
    # Intel SYCL uses libiomp5 for OpenMP
    find_library(OpenMP_libiomp5_LIBRARY
        NAMES iomp5
        HINTS ENV INTEL_ONEAPI_ROOT
        PATH_SUFFIXES compiler/latest/lib
                      compiler/latest/linux/compiler/lib/intel64_lin
    )
endif()

# =============================================================================
# Create targets for each backend
# =============================================================================
set(ALL_BENCH_TARGETS "")

foreach(BACKEND ${BACKEND_LIST})
    string(TOUPPER "${BACKEND}" BACKEND_UPPER)
    string(TOLOWER "${BACKEND}" BACKEND_LOWER)
    set(TARGET_NAME "bench_sim_${BACKEND_LOWER}")
    
    message(STATUS "Configuring target: ${TARGET_NAME}")

    # --- CUDA Backend ---
    if(BACKEND_UPPER STREQUAL "CUDA")
        add_executable(${TARGET_NAME} run.cpp ${SIMULATION_SOURCES} ${CUDA_SOURCES})
        target_link_libraries(${TARGET_NAME} PRIVATE
            CUDA::cudart
            OpenMP::OpenMP_CXX
            Eigen3::Eigen
        )
        target_include_directories(${TARGET_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
        set_target_properties(${TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    # --- SYCL Backend ---
    elseif(BACKEND_UPPER STREQUAL "SYCL")
        add_executable(${TARGET_NAME} run.cpp ${SIMULATION_SOURCES})
        target_link_libraries(${TARGET_NAME} PRIVATE
            sycl
            ${OpenMP_libiomp5_LIBRARY}
            Eigen3::Eigen
        )
        target_compile_options(${TARGET_NAME} PRIVATE -fsycl -qopenmp)

    # --- CPU Backend ---
    else()
        add_executable(${TARGET_NAME} run.cpp ${SIMULATION_SOURCES})
        target_link_libraries(${TARGET_NAME} PRIVATE
            OpenMP::OpenMP_CXX
            Eigen3::Eigen
        )
    endif()

    # Common properties for all backends
    target_include_directories(${TARGET_NAME} PRIVATE
        ${PLANTSIM_SRC_DIR}
        ${PLANTSIM_INCLUDE_DIR}
    )
    target_compile_definitions(${TARGET_NAME} PRIVATE
        TARGET_BACKEND="${BACKEND_LOWER}"
        BACKEND_${BACKEND_UPPER}=1
    )
    set_target_properties(${TARGET_NAME} PROPERTIES
        CXX_STANDARD 23
        CXX_STANDARD_REQUIRED ON
    )

    list(APPEND ALL_BENCH_TARGETS ${TARGET_NAME})
endforeach()

# Convenience target to build all backends
add_custom_target(bench_sim DEPENDS ${ALL_BENCH_TARGETS})


