
find_package(IntelSYCL REQUIRED)

add_executable(sycl_test main.cpp)

target_link_libraries(sycl_test PRIVATE sycl)

target_compile_options(sycl_test PRIVATE -fsycl)
