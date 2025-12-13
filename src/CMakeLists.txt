
set(BINARY ${CMAKE_PROJECT_NAME})

# =============================================================================
# Common dependencies
# =============================================================================
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(imgui REQUIRED)
find_package(Freetype REQUIRED)
find_package(OpenMP REQUIRED)

if(BACKEND_UPPER STREQUAL "CPU")
    find_package(Eigen3 REQUIRED NO_MODULE)
elseif(BACKEND_UPPER STREQUAL "CUDA")
    find_package(Eigen3 REQUIRED NO_MODULE)
    find_package(CUDAToolkit REQUIRED)
elseif(BACKEND_UPPER STREQUAL "SYCL")
    find_package(Eigen3 REQUIRED NO_MODULE)
    find_package(IntelSYCL REQUIRED)
    find_library(OpenMP_libiomp5_LIBRARY
        NAMES iomp5
        HINTS ENV INTEL_ONEAPI_ROOT
        PATH_SUFFIXES compiler/latest/lib
                      compiler/latest/linux/compiler/lib/intel64_lin
    )
endif()

file(GLOB_RECURSE SOURCES LIST_DIRECTORIES false *.h *.cpp *.c *.cu)
set(TEST_SOURCES ${SOURCES})
list(FILTER TEST_SOURCES EXCLUDE REGEX ".*main.cpp$")

set(COMMON_LIBS
    OpenGL::GL
    glfw
    glm::glm
    imgui
    Freetype::Freetype
)

if(BACKEND_UPPER STREQUAL "CPU")
    set(BACKEND_LIBS
        OpenMP::OpenMP_CXX
        Eigen3::Eigen
    )
elseif(BACKEND_UPPER STREQUAL "CUDA")
    set(BACKEND_LIBS
        CUDA::cudart
        OpenMP::OpenMP_CXX
        Eigen3::Eigen
    )
elseif(BACKEND_UPPER STREQUAL "SYCL")
    set(BACKEND_LIBS
        sycl
        ${OpenMP_libiomp5_LIBRARY}
        Eigen3::Eigen
    )
else()
    set(BACKEND_LIBS OpenMP::OpenMP_CXX)
endif()

add_executable(${BINARY}_run ${SOURCES})
target_include_directories(${BINARY}_run PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(${BINARY}_run PRIVATE ${COMMON_LIBS} ${BACKEND_LIBS})

add_library(${BINARY}_lib STATIC ${TEST_SOURCES})
target_include_directories(${BINARY}_lib PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(${BINARY}_lib PUBLIC ${COMMON_LIBS} ${BACKEND_LIBS})

if(Boost_FOUND)
    target_link_libraries(${BINARY}_run ${Boost_LIBRARIES})
endif()

target_compile_definitions(${BINARY}_run PRIVATE
    TARGET_BACKEND="${BACKEND_LOWER}"
    BACKEND_${BACKEND_UPPER}=1
)
