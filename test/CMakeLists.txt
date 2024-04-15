project(tests)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)

enable_testing()

set(BINARY ${CMAKE_PROJECT_NAME}_tests)
file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES true *.h *.cpp)
set(SOURCES ${TEST_SOURCES})
add_executable(${BINARY} ${TEST_SOURCES} ../include/utils.h)
target_link_libraries(${BINARY} ${CMAKE_PROJECT_NAME}_lib gtest gtest_main)

include(GoogleTest)
gtest_discover_tests(${BINARY})