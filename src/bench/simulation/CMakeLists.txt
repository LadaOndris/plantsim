message(STATUS "Building simulation benchmarks with backends: ${BACKEND_LIST}")

set(ALL_BENCH_TARGETS "")

foreach(BACKEND ${BACKEND_LIST})
    string(TOUPPER "${BACKEND}" BACKEND_UPPER)
    string(TOLOWER "${BACKEND}" BACKEND_LOWER)
    set(TARGET_NAME bench_sim_${BACKEND_LOWER})

    add_executable(${TARGET_NAME} run.cpp)

    target_link_libraries(${TARGET_NAME} PRIVATE
        plantsim_simulation_${BACKEND_LOWER}
    )

    if(BACKEND_UPPER STREQUAL "CUDA")
        set_target_properties(${TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    endif()

    list(APPEND ALL_BENCH_TARGETS ${TARGET_NAME})
    message(STATUS "Configured ${TARGET_NAME}")
endforeach()

add_custom_target(bench_sim DEPENDS ${ALL_BENCH_TARGETS})
