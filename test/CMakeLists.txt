project(tests)

# Dependencies

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

find_package(Eigen3 REQUIRED NO_MODULE)

# Sources

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src)

set(INCLUDE_DIR ${SRC_DIR})

# All the sources being tested: simulation and plants for now.
set(SOURCES
    ${SRC_DIR}/simulation/cpu/Photosynthesis.cpp
    ${SRC_DIR}/simulation/cpu/LightComputation.cpp
    ${SRC_DIR}/simulation/cpu/SoilAbsorption.cpp
    ${SRC_DIR}/simulation/cpu/GridShiftHelper.cpp
)

set(TEST_SOURCES
    #${CMAKE_CURRENT_SOURCE_DIR}/simulation/GpuContextTests.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/simulation/ResourceTransferTests.cpp  # Temporarily disabled - has API issues
    ${CMAKE_CURRENT_SOURCE_DIR}/simulation/GridTopologyTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/simulation/MapPrinterTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/simulation/PhotosynthesisTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/simulation/LightComputationTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/simulation/LightPhotosynthesisIntegrationTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/simulation/SoilAbsorptionTests.cpp
)

set(BINARY_NAME plantsim_tests)


add_executable(${BINARY_NAME} ${TEST_SOURCES} ${SOURCES} ../include/utils.h)

target_link_libraries(${BINARY_NAME}
    gtest
    gtest_main
    Eigen3::Eigen
)

target_include_directories(${BINARY_NAME} PRIVATE
    ${INCLUDE_DIR}
    ${gtest_SOURCE_DIR}/include
)
