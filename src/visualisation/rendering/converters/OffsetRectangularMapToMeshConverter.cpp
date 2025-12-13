
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <glm/vec3.hpp>
#include "OffsetRectangularMapToMeshConverter.h"

MeshData OffsetRectangularMapToMeshConverter::convert(const GridTopology &topology) const {
    // Todo: Remove duplicate code between converters
    auto width = topology.width;
    auto height = topology.height;
    std::cout << "[Map] Width: " << width << ", height: " << height << std::endl;
    assert (width == height);

    double cellCentersDistance = 1 / static_cast<double>(width - 1);
    double cellRadius = cellCentersDistance / 2;
    double triangleHeight = sqrt(3) / 2 * cellRadius;

    MeshData meshData{};

    constexpr int indicesPerCell = 18;
    constexpr int verticesPerCell = 7;
    std::vector<unsigned int> singleCellIndices(indicesPerCell); // 6 triangles * 3 vertices = 18
    for (int i = 0; i < 6; ++i) {
        singleCellIndices[i * 3 + 0] = 0; // The center of the cell
        singleCellIndices[i * 3 + 1] = i + 1; // First vertex on the boundary
        singleCellIndices[i * 3 + 2] = (i + 1) % 6 + 1; // Second vertex on the boundary
    }

    float heightFloat = static_cast<float>(height);
    float widthFloat = static_cast<float>(width);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            auto currentCellVertexIndices = &meshData.cellVerticesMap[std::make_pair(i, j)];
            std::vector<unsigned int> currentCellIndices{singleCellIndices};
            // Shift the base cell indices to form indices for the current cell.
            std::transform(currentCellIndices.begin(), currentCellIndices.end(), currentCellIndices.begin(),
                           [i, j, height](auto &item) {
                               return item + (i * height + j) * verticesPerCell;
                           });
            meshData.indices.insert(meshData.indices.end(), currentCellIndices.begin(), currentCellIndices.end());

            float maxY = (heightFloat) * 2 * triangleHeight;
            float maxX = (widthFloat) * 3 / 2 * cellRadius + cellRadius;
            float centerY = static_cast<float>(i) * 2 * triangleHeight / maxY + triangleHeight;
            float centerX = static_cast<float>(j) * 3 / 2 * cellRadius / maxX + cellRadius;

            // Shift the center for specific odd columns
            if (j % 2 == 1) {
                centerY -= triangleHeight;
            }

            currentCellVertexIndices->push_back(meshData.vertices.size());
            meshData.vertices.push_back(GLVertex{{centerX, centerY, 0}});

            for (int k = 0; k < 6; ++k) {
                float x = centerX + cellRadius * cos(k * (2.0f * M_PI) / 6.0f);
                float y = centerY + cellRadius * sin(k * (2.0f * M_PI) / 6.0f);

                currentCellVertexIndices->push_back(meshData.vertices.size());
                meshData.vertices.push_back(GLVertex{x, y, 0});
            }
        }
    }
    return meshData;
}
