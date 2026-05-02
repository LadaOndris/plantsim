#pragma once

#include <cstddef>
#include <vector>
#include "visualisation/rendering/GLVertex.h"

inline constexpr std::size_t VERTICES_PER_CELL = 7;
inline constexpr std::size_t INDICES_PER_CELL = 18;

constexpr std::size_t cellVertexBaseIndex(int row, int col, int width) noexcept {
    return (static_cast<std::size_t>(row) * static_cast<std::size_t>(width)
          + static_cast<std::size_t>(col)) * VERTICES_PER_CELL;
}

struct MeshData {
    std::vector<GLVertex> vertices;
    std::vector<unsigned int> indices;
};
