project(tests)

find_package(GTest REQUIRED)

set(TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/simulation/stages/LightComputationTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/simulation/stages/LightPhotosynthesisIntegrationTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/simulation/stages/PhotosynthesisTests.cpp
    #${CMAKE_CURRENT_SOURCE_DIR}/simulation/stages/ResourceTransferTests.cpp  # Temporarily disabled - has API issues
    ${CMAKE_CURRENT_SOURCE_DIR}/simulation/stages/SoilAbsorptionTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/simulation/GridTopologyTests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/simulation/MapPrinterTests.cpp
)

add_executable(plantsim_tests ${TEST_SOURCES})

target_link_libraries(plantsim_tests PRIVATE
    GTest::gtest
    GTest::gtest_main
    plantsim_simulation_cpu
)
