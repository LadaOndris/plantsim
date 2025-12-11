cmake_minimum_required(VERSION 3.27)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_subdirectory(simulation)

if(TARGET_BACKEND STREQUAL "SYCL")
    add_subdirectory(sycltest)
endif()
