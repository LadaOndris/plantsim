cmake_minimum_required(VERSION 3.27)

find_package(IntelSYCL REQUIRED)
find_package (Eigen3 REQUIRED NO_MODULE)

# Manual OpenMP configuration for Intel SYCL compiler
# Uses Intel's libiomp5 implementation
set(OpenMP_CXX_FLAGS "-qopenmp")
set(OpenMP_CXX_LIB_NAMES "libiomp5")
# Find libiomp5 in Intel oneAPI installation
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

set(BENCH_SIMULATION_BINARY bench_simulation)

file(GLOB_RECURSE SIMULATION_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/simulation/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/plants/*.cpp"
)

add_executable(${BENCH_SIMULATION_BINARY} run.cpp ${SIMULATION_SOURCES})

target_link_libraries(${BENCH_SIMULATION_BINARY} PRIVATE 
    sycl 
    ${OpenMP_libiomp5_LIBRARY}
    Eigen3::Eigen
)

target_include_directories(${BENCH_SIMULATION_BINARY} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    "${SYCL_INCLUDE_DIR}"
    "${EIGEN3_INCLUDE_DIR}"
)

set_target_properties(${BENCH_SIMULATION_BINARY} PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

target_compile_options(${BENCH_SIMULATION_BINARY} PRIVATE -fsycl ${OpenMP_CXX_FLAGS})
