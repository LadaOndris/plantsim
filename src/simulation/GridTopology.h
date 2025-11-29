
#pragma once

#include <array>
#include <cstddef>

enum class NeighborOffset {
    Right,
    TopRight,
    TopLeft,
    Left,
    BottomLeft,
    BottomRight
};

/**
 * @brief Defines the static structure of a hexagonal grid using axial coordinates.
 *
 * @details The map is represented in **axial** coordinates for hexagonal grids.
 * The storage is a 2D array. Since the actual hexagonal map is a rectangle, the 
 * underlying storage contains unused cells.
 * 
 */
class GridTopology {
public:
    const int width;
    const int height;

    static constexpr std::pair<int, int> getNeighborOffset(NeighborOffset direction) {
        return neighborOffsets[static_cast<int>(direction)];
    }

    // The 6 possible neighbor offsets in axial coordinates
    static constexpr std::array<std::pair<int, int>, 6> neighborOffsets {
        { {+1, 0}, {+1, -1}, {0, -1},
          {-1, 0}, {-1, +1}, {0, +1} }
    };

    constexpr GridTopology(int width, int height) 
    : width(width), height(height) {}

    [[nodiscard]] constexpr size_t totalCells() const {
        return static_cast<size_t>(width) * height;
    }

    /**
     * @brief Returns the storage dimensions required for the axial coordinate map.
     * 
     * @details The axial coordinates are stored in a 2D array with unused cells.
     * The storage width is larger than the logical width to accommodate the
     * hexagonal stagger pattern.
     * 
     * @see https://www.redblobgames.com/grids/hexagons/#map-storage
     * 
     * @return Pair of (storageWidth, storageHeight)
     */
    [[nodiscard]] constexpr std::pair<int, int> getStorageDimension() const {
        int additionalWidth = (height - 1) / 2;
        return {width + additionalWidth, height};
    }

    /**
     * @brief Converts axial coordinates (q, r) to storage coordinates (col, row).
     * 
     * @details For an axial map of size 7x7:
     * - [q,r] = [0,0] maps to storage[3,0] (offset by additionalWidth)
     * - [q,r] = [-3,6] maps to storage[0,6]
     * - [q,r] = [6,1] maps to storage[6,1]
     * 
     * The conversion applies an offset based on the row to account for the
     * hexagonal stagger: col = q + (height - 1) / 2, row = r
     * 
     * @param axial The axial coordinates as (q, r)
     * @return Storage coordinates as (col, row)
     */
    [[nodiscard]] constexpr std::pair<int, int> axialToStorageCoord(std::pair<int, int> axial) const {
        auto [q, r] = axial;
        int offset = (height - 1) / 2;
        int col = q + offset;
        int row = r;
        return {col, row};
    }

    /**
     * @brief Converts storage coordinates (col, row) to axial coordinates (q, r).
     * 
     * @details Inverse of axialToStorageCoord.
     * 
     * @param storage The storage coordinates as (col, row)
     * @return Axial coordinates as (q, r)
     */
    [[nodiscard]] constexpr std::pair<int, int> storageToAxialCoord(std::pair<int, int> storage) const {
        auto [col, row] = storage;
        int offset = (height - 1) / 2;
        int q = col - offset;
        int r = row;
        return {q, r};
    }

    /**
     * @brief Converts axial coordinates to a linear storage index.
     * 
     * @param r The r coordinate (row in axial)
     * @param q The q coordinate (column in axial)
     * @return Linear index into the storage array
     */
    [[nodiscard]] constexpr int toIndex(int r, int q) const {
        auto [col, row] = axialToStorageCoord({q, r});
        auto [storageWidth, storageHeight] = getStorageDimension();
        return row * storageWidth + col;
    }

    /**
     * @brief Checks if axial coordinates are within valid storage bounds.
     * 
     * @param r The r coordinate (row in axial)
     * @param q The q coordinate (column in axial)
     * @return True if coordinates map to valid storage location
     */
    [[nodiscard]] constexpr bool isValid(int r, int q) const {
        auto [col, row] = axialToStorageCoord({q, r});
        auto [storageWidth, storageHeight] = getStorageDimension();
        return row >= 0 && row < storageHeight && col >= 0 && col < storageWidth;
    }
};