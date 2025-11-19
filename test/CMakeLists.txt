project(tests)

# Dependencies

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
FetchContent_MakeAvailable(googletest)

# Sources

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src)

set(INCLUDE_DIR ${SRC_DIR})

# All the sources being tested: simulation and plants for now.
set(SOURCES 
    ${SRC_DIR}/simulation/ <FILL>
    ${SRC_DIR}/plants/ <FILL>
)

set(TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/simulation/GpuContextTests.cpp
)

set(BINARY_NAME plantsim_tests)


add_executable(${BINARY_NAME} ${TEST_SOURCES} ../include/utils.h)

target_link_libraries(${BINARY_NAME} 
    gtest 
    gtest_main
)

target_include_directories(${BINARY_NAME} PRIVATE
    ${INCLUDE_DIR}
    ${gtest_SOURCE_DIR}/include
)
