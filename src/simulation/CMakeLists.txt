foreach(BACKEND ${BACKEND_LIST})
    string(TOUPPER "${BACKEND}" BACKEND_UPPER)
    string(TOLOWER "${BACKEND}" BACKEND_LOWER)

    set(SIM_LIB plantsim_simulation_${BACKEND_LOWER})

    set(SIMULATION_SOURCES
        Options.cpp
        SimulatorFactory.cpp
    )

    if(BACKEND_UPPER STREQUAL "CPU")
        find_package(Eigen3 REQUIRED NO_MODULE)
        list(APPEND SIMULATION_SOURCES
            cpu/CpuSimulator.cpp
            cpu/stages/DeadDecay.cpp
            cpu/stages/LightComputation.cpp
            cpu/stages/MaintenanceAndDeath.cpp
            cpu/stages/Photosynthesis.cpp
            cpu/stages/RandomNeighborReproduction.cpp
            cpu/stages/ResourceTransfer.cpp
            cpu/stages/SoilAbsorption.cpp
            cpu/stages/SoilDiffusion.cpp
            cpu/utils/GridShiftHelper.cpp
        )

    elseif(BACKEND_UPPER STREQUAL "CUDA")
        if(NOT CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
            set(CMAKE_CUDA_ARCHITECTURES "61;86" CACHE STRING "CUDA architectures" FORCE)
        endif()
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
        list(APPEND SIMULATION_SOURCES
            cuda/CudaSimulator.cu
            cuda/CudaState.cu
            cuda/RandomNeighborReproduction.cu
            cuda/ResourceTransfer.cu
        )

    elseif(BACKEND_UPPER STREQUAL "SYCL")
        find_package(IntelSYCL REQUIRED)
        find_library(OpenMP_libiomp5_LIBRARY
            NAMES iomp5
            HINTS ENV INTEL_ONEAPI_ROOT
            PATH_SUFFIXES compiler/latest/lib
                          compiler/latest/linux/compiler/lib/intel64_lin
        )
    endif()

    add_library(${SIM_LIB} STATIC ${SIMULATION_SOURCES})

    target_link_libraries(${SIM_LIB} PUBLIC plantsim_common)

    target_compile_definitions(${SIM_LIB} PUBLIC
        TARGET_BACKEND="${BACKEND_LOWER}"
        BACKEND_${BACKEND_UPPER}=1
    )

    if(BACKEND_UPPER STREQUAL "CPU")
        target_link_libraries(${SIM_LIB} PUBLIC Eigen3::Eigen)

    elseif(BACKEND_UPPER STREQUAL "CUDA")
        target_link_libraries(${SIM_LIB} PUBLIC CUDA::cudart)
        set_target_properties(${SIM_LIB} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    elseif(BACKEND_UPPER STREQUAL "SYCL")
        target_link_libraries(${SIM_LIB} PUBLIC sycl ${OpenMP_libiomp5_LIBRARY})
        target_compile_options(${SIM_LIB} PRIVATE -fsycl -qopenmp)
    endif()

    message(STATUS "Configured ${SIM_LIB}")
endforeach()
