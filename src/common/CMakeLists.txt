add_library(plantsim_common STATIC
    Random.cpp
    Random.h
    ApplicationConfig.h
    NotImplementedException.h
    utils.h
)

target_include_directories(plantsim_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)
