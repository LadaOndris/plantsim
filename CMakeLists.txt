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

add_executable(plantsim src/main.cpp src/Environment.cpp src/Environment.h src/WorldState.cpp src/WorldState.h src/Constraint.cpp src/Constraint.h src/Point.cpp src/Point.h src/Air.cpp src/Air.h src/Soil.cpp src/Soil.h src/Cell.cpp src/Cell.h src/genetics/Chromosome.cpp src/genetics/Chromosome.h src/genetics/Gene.cpp src/genetics/Gene.h)

if(Boost_FOUND)
    target_link_libraries(plantsim ${Boost_LIBRARIES})
endif()