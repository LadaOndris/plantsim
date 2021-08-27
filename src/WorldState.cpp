//
// Created by lada on 8/26/21.
//

#include "WorldState.h"

WorldState::WorldState(int width, int height) :
        width(width), height(height) {
    std::vector<std::unique_ptr<Process>> processes;

    entity = std::make_unique<Entity>(processes);
    //scoped_array mapTemp(new Point[width * height]);
    //map.swap(mapTemp);
}
