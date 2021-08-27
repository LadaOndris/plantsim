cmake_minimum_required(VERSION 3.0)
project(plantsim)

set(CMAKE_CXX_STANDARD 14)
set(BINARY ${CMAKE_PROJECT_NAME})

find_package(Boost 1.57.0 COMPONENTS system filesystem REQUIRED)

if(Boost_FOUND)
    message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
    message(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")
    message(STATUS "Boost_VERSION: ${Boost_VERSION}")

    include_directories(${Boost_INCLUDE_DIRS})
endif()

file(GLOB_RECURSE SOURCES LIST_DIRECTORIES true *.h *.cpp)
set(SOURCES ${SOURCES})
add_executable(${BINARY}_run ${SOURCES})
# add_executable(plantsim main.cpp Environment.cpp Environment.h WorldState.cpp WorldState.h Constraint.cpp Constraint.h Point.cpp Point.h Air.cpp Air.h Soil.cpp Soil.h Cell.cpp Cell.h genetics/Chromosome.cpp genetics/Chromosome.h genetics/Gene.cpp genetics/Gene.h)
add_library(${BINARY}_lib STATIC ${SOURCES})

if(Boost_FOUND)
    target_link_libraries(${BINARY}_run ${Boost_LIBRARIES})
endif()
