cmake_minimum_required(VERSION 3.27)

# Find SYCL package
find_package(IntelSYCL REQUIRED)

# Create the executable for the SYCL test
add_executable(sycl_test main.cpp)

# Link against SYCL libraries
target_link_libraries(sycl_test PRIVATE sycl)

target_include_directories(sycl_test PRIVATE "${SYCL_INCLUDE_DIR}")

# Set C++ standard for this target
set_target_properties(sycl_test PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Optional: Add SYCL compiler flags if needed
target_compile_options(sycl_test PRIVATE -fsycl)
