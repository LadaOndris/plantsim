#include "resources/ResourceLoader.h"
#include <cmrc/cmrc.hpp>

CMRC_DECLARE(plantsim_shaders);

namespace resources {
    std::string load(const std::string& path) {
        auto fs = cmrc::plantsim_shaders::get_filesystem();
        auto file = fs.open(path);
        return std::string(file.begin(), file.end());
    }
}
