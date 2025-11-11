set(BENCH_SIMULATION_BINARY bench_simulation)

find_package(OpenMP REQUIRED)

file(GLOB_RECURSE SIMULATION_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/simulation/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/plants/*.cpp"
)

add_executable(${BENCH_SIMULATION_BINARY} run.cpp ${SIMULATION_SOURCES})

target_include_directories(${BENCH_SIMULATION_BINARY} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(${BENCH_SIMULATION_BINARY} PRIVATE
    OpenMP::OpenMP_CXX
)
