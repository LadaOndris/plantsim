
#pragma once

#include <array>
#include <cstddef>
#include <vector>

enum class NeighborOffset {
    Right,
    TopRight,
    TopLeft,
    Left,
    BottomLeft,
    BottomRight
};

struct AxialCoord { 
    int q; 
    int r; 

    constexpr int asFlat(const AxialCoord& dim) const {
        return r * dim.q + q;
    }
    
    constexpr bool operator==(const AxialCoord& other) const {
        return q == other.q && r == other.r;
    }

    constexpr AxialCoord operator+(const AxialCoord& other) const noexcept {
        return AxialCoord{q + other.q, r + other.r};
    }

    constexpr AxialCoord& operator+=(const AxialCoord& other) noexcept {
        q += other.q;
        r += other.r;
        return *this;
    }

    constexpr AxialCoord operator-(const AxialCoord& other) const noexcept {
        return AxialCoord{q - other.q, r - other.r};
    }

    constexpr AxialCoord& operator-=(const AxialCoord& other) noexcept {
        q -= other.q;
        r -= other.r;
        return *this;
    }
};

struct OffsetCoord { int col; int row; };

struct StorageCoord { 
    int x; 
    int y;

    constexpr int asFlat(const StorageCoord& dim) const {
        return y * dim.x + x;
    }
    
    constexpr bool operator==(const StorageCoord& other) const {
        return x == other.x && y == other.y;
    }

    constexpr StorageCoord operator+(const StorageCoord& other) const noexcept {
        return StorageCoord{x + other.x, y + other.y};
    }

    constexpr StorageCoord& operator+=(const StorageCoord& other) noexcept {
        x += other.x;
        y += other.y;
        return *this;
    }
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

    static constexpr AxialCoord getNeighborOffset(NeighborOffset direction) {
        return neighborOffsets[static_cast<int>(direction)];
    }

    // The 6 possible neighbor offsets in axial coordinates
    static constexpr std::array<AxialCoord, 6> neighborOffsets {
        AxialCoord{+1, 0}, AxialCoord{+1, -1}, AxialCoord{0, -1},
        AxialCoord{-1, 0}, AxialCoord{-1, +1}, AxialCoord{0, +1}
    };

    constexpr GridTopology(int width, int height) 
    : width(width), height(height) {}

    [[nodiscard]] constexpr size_t totalCells() const {
        return static_cast<size_t>(width) * height;
    }

    [[nodiscard]] constexpr AxialCoord getDimension() const {
        return AxialCoord{width, height};
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
    [[nodiscard]] constexpr StorageCoord getStorageDimension() const {
        int additionalWidth = (height - 1) / 2;
        return {width + additionalWidth, height};
    }

    /**
     * @brief Converts axial coordinates (q, r) to storage coordinates (x, y).
     * 
     * @details For an axial map of size 7x7:
     * - [q,r] = [0,0] maps to storage[3,0] (offset by additionalWidth)
     * - [q,r] = [-3,6] maps to storage[0,6]
     * - [q,r] = [6,1] maps to storage[9,1]
     * 
     * The conversion applies an offset based on the row to account for the
     * hexagonal stagger: x = q + (height - 1) / 2, y = r
     * 
     * @param axial The axial coordinates
     * @return Storage coordinates
     */
    [[nodiscard]] constexpr StorageCoord axialToStorageCoord(AxialCoord axial) const {
        int offset = (height - 1) / 2;
        int x = axial.q + offset;
        int y = axial.r;
        return {x, y};
    }

    /**
     * @brief Converts storage coordinates (x, y) to axial coordinates (q, r).
     * 
     * @details Inverse of axialToStorageCoord.
     * 
     * @param storage The storage coordinates
     * @return Axial coordinates
     */
    [[nodiscard]] constexpr AxialCoord storageToAxialCoord(StorageCoord storage) const {
        int offset = (height - 1) / 2;
        int q = storage.x - offset;
        int r = storage.y;
        return {q, r};
    }

    /**
     * @brief Converts axial coordinates to a linear storage index.
     * 
     * @param axial The axial coordinates
     * @return Linear index into the storage array
     */
    [[nodiscard]] constexpr int toIndex(AxialCoord axial) const {
        StorageCoord storage = axialToStorageCoord(axial);
        StorageCoord dim = getStorageDimension();
        return storage.y * dim.x + storage.x;
    }

    /**
     * @brief Checks if axial coordinates represent a valid hex cell (not padding).
     * 
     * @details In the rectangular storage, some cells are padding/dummy cells
     * that don't correspond to actual hex cells. This function returns true
     * only for coordinates that map to real hex cells within the logical
     * width x height grid.
     * 
     * For a hex grid, valid axial coordinates satisfy:
     * - 0 <= r < height
     * - For each row r, q ranges based on the hex stagger pattern
     * 
     * @param axial The axial coordinates
     * @return True if coordinates represent a valid hex cell
     */
    [[nodiscard]] constexpr bool isValid(AxialCoord axial) const {
        // Check row bounds
        if (axial.r < 0 || axial.r >= height) {
            return false;
        }
        
        // For odd-r offset layout, calculate valid q range for this row
        // The minimum q shifts left on odd rows
        int parity = axial.r & 1;
        int minQ = -(axial.r - parity) / 2;
        int maxQ = minQ + width - 1;
        
        return axial.q >= minQ && axial.q <= maxQ;
    }

    [[nodiscard]] constexpr bool isValid(StorageCoord storage) const {
        // Quick bounds check against underlying storage dimensions
        StorageCoord dim = getStorageDimension();
        if (storage.x < 0 || storage.x >= dim.x || storage.y < 0 || storage.y >= dim.y) {
            return false;
        }

        int offset = (height - 1) / 2;
        int parity = storage.y & 1;
        int minStorageX = offset - ((storage.y - parity) / 2);
        int maxStorageX = minStorageX + width - 1;

        return storage.x >= minStorageX && storage.x <= maxStorageX;
    }
};


inline OffsetCoord axialToOddr(const AxialCoord& hex) {
    int parity = hex.r & 1;
    int col = hex.q + (hex.r - parity) / 2;
    int row = hex.r;
    return OffsetCoord{col, row};
}

inline AxialCoord oddrToAxial(const OffsetCoord& hex) {
    int parity = hex.row & 1;
    int q = hex.col - (hex.row - parity) / 2;
    int r = hex.row;
    return AxialCoord{q, r};
}

inline std::vector<int> store(std::vector<int> data, int width, int height, int defaultFillValue = -1) {
    std::vector<int> storage;
    GridTopology topology(width, height);
    StorageCoord dim = topology.getStorageDimension();
    storage.resize(dim.x * dim.y, defaultFillValue);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int value = data[y * width + x];
            AxialCoord axial = oddrToAxial({x, y});
            StorageCoord storageCoord = topology.axialToStorageCoord(axial);
            storage[storageCoord.y * dim.x + storageCoord.x] = value;
        }
    }

    return storage;
}
