
include(FetchContent)

FetchContent_Declare(
    cmrc
    GIT_REPOSITORY https://github.com/vector-of-bool/cmrc.git
    GIT_TAG        master
)

FetchContent_MakeAvailable(cmrc)
