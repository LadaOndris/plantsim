cmake_minimum_required(VERSION 3.0)
project(plantsim)

set(CMAKE_CXX_STANDARD 14)

find_package(Boost 1.57.0 COMPONENTS system filesystem REQUIRED)

if(Boost_FOUND)
    message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
    message(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")
    message(STATUS "Boost_VERSION: ${Boost_VERSION}")

    include_directories(${Boost_INCLUDE_DIRS})
endif()

add_executable(plantsim main.cpp Environment.cpp Environment.h WorldState.cpp WorldState.h Constraint.cpp Constraint.h Point.cpp Point.h)

if(Boost_FOUND)
    target_link_libraries(plantsim ${Boost_LIBRARIES})
endif()