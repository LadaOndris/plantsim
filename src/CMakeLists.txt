add_subdirectory(common)
add_subdirectory(simulation)

if(BUILD_VISUALIZATION)
    add_subdirectory(resources)
    add_subdirectory(visualisation)

    foreach(BACKEND ${BACKEND_LIST})
        string(TOUPPER "${BACKEND}" BACKEND_UPPER)
        string(TOLOWER "${BACKEND}" BACKEND_LOWER)

        add_executable(plantsim_${BACKEND_LOWER} main.cpp)

        set_target_properties(plantsim_${BACKEND_LOWER} PROPERTIES
            OUTPUT_NAME plantsim
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/${BACKEND_LOWER}
        )

        target_link_libraries(plantsim_${BACKEND_LOWER} PRIVATE
            plantsim_simulation_${BACKEND_LOWER}
            plantsim_visualisation
            plantsim_resources
        )

        message(STATUS "Configured plantsim_${BACKEND_LOWER}")
    endforeach()
endif()

if(BUILD_BENCH)
    add_subdirectory(bench)
endif()

if(BUILD_EXPERIMENTS)
    add_subdirectory(experiments)
endif()
