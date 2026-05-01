include(${CMAKE_SOURCE_DIR}/third-party/cmrc/cmrc.cmake)

cmrc_add_resource_library(plantsim_shaders
    ALIAS plantsim::shaders
    WHENCE ${CMAKE_SOURCE_DIR}/src/visualisation
    ${CMAKE_SOURCE_DIR}/src/visualisation/shaders/map/shader.vert
    ${CMAKE_SOURCE_DIR}/src/visualisation/shaders/map/shader.frag
)

add_library(plantsim_resources STATIC
    ResourceLoader.cpp
    ResourceLoader.h
)

target_link_libraries(plantsim_resources PUBLIC
    plantsim_common
    plantsim::shaders
)
