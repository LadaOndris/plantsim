#pragma once

#include <vector>
#include <unordered_map>
#include "visualisation/rendering/GLVertex.h"

struct PairHash {
    template<typename T1, typename T2>
    std::size_t operator()(const std::pair<T1, T2> &pair) const {
        // Combine the hash values of the pair's elements
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};

struct MeshData {
    std::vector<GLVertex> vertices;
    std::vector<unsigned int> indices;

    // We need to keep track of which vertices correspond to which cell on the map.
    std::unordered_map<std::pair<int, int>, std::vector<size_t>, PairHash> cellVerticesMap{};
};

