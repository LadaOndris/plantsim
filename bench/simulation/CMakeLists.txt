cmake_minimum_required(VERSION 3.27)

find_package(IntelSYCL REQUIRED)

set(BENCH_SIMULATION_BINARY bench_simulation)

file(GLOB_RECURSE SIMULATION_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/simulation/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/plants/*.cpp"
)

add_executable(${BENCH_SIMULATION_BINARY} run.cpp ${SIMULATION_SOURCES})

target_link_libraries(${BENCH_SIMULATION_BINARY} PRIVATE sycl)

target_include_directories(${BENCH_SIMULATION_BINARY} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    "${SYCL_INCLUDE_DIR}"
)

set_target_properties(${BENCH_SIMULATION_BINARY} PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

target_compile_options(${BENCH_SIMULATION_BINARY} PRIVATE -fsycl)
