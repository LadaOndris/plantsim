//
// Created by lada on 8/26/21.
//

#include "WorldState.h"

WorldState::WorldState(int width, int height) {
    scoped_array mapTemp(new Point[width * height]);
    map.swap(mapTemp);
}
