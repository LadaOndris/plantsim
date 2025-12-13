#pragma once

#include <vector>
#include <array>
#include <cassert>
#include <span>
#include "Map.h"
#include "Point.h"

/**
 * Template-based axial rectangular map that stores cells of type CellData.
 * Uses array-of-structs (AoS) layout for improved cache locality.
 * Each cell is of type CellData, containing application-specific data.
 *
 * The map uses axial coordinates (q, r) and includes padding for SIMD operations.
 *
 * @tparam CellData The type stored in each cell (e.g., CellState)
 */
template<typename CellData>
class AxialRectangularMap {

public:
    /**
     * Initializes the storage of the map.
     * It uses the axial coordinates (q, r).
     *
     * @param width The maximum q coordinate.
     * @param height The maximum r coordinate.
     */
    explicit AxialRectangularMap(int width, int height);

    [[nodiscard]] constexpr int getWidth() const noexcept;

    [[nodiscard]] constexpr int getHeight() const noexcept;

    [[nodiscard]] constexpr std::pair<int, int> getMaxCoords() const noexcept;

    [[nodiscard]] constexpr std::pair<int, int> getStorageDims() const noexcept;

    /**
     * Gets the linear storage index for the given axial coordinates.
     * @param r Row coordinate
     * @param q Column coordinate
     * @return Linear index into the storage array
     */
    [[nodiscard]] constexpr int getStorageCoord(int r, int q) const noexcept {
        // Storage includes padding: (storageDims.first + 2) x (storageDims.second + 2)
        // Offset by 1 to account for padding on left/top
        return (r + 1) * (storageDims.first + 2) + q + 1;
    }

    /**
     * Gets the validity mask coordinate for the given axial coordinates.
     * @param r Row coordinate
     * @param q Column coordinate
     * @return Linear index into the validity mask array
     */
    [[nodiscard]] constexpr int getValidityMaskCoord(int r, int q) const noexcept {
        // Offset of one point on each side
        return (r + 1) * (storageDims.first + 2) + q + 1;
    }

    /**
     * Gets the cell data at the given linear storage index.
     * @param idx Linear storage index
     * @return Reference to the cell data
     */
    CellData &getCellAt(int idx) noexcept {
        return cells[idx];
    }

    const CellData &getCellAt(int idx) const noexcept {
        return cells[idx];
    }

    /**
     * Gets mutable reference to all cells (array-of-structs layout).
     * @return Reference to the cells vector
     */
    std::vector<CellData> &getCells() noexcept {
        return cells;
    }

    CellData &getCellAt(int r, int q) noexcept {
        return cells[getStorageCoord(r, q)];
    }

    /**
     * Gets const reference to all cells (array-of-structs layout).
     * @return Const reference to the cells vector
     */
    const std::vector<CellData> &getCells() const noexcept {
        return cells;
    }

    /**
     * Checks if a cell at the given coordinates is valid (within hexagonal bounds).
     * @param r Row coordinate
     * @param q Column coordinate
     * @return true if the cell is valid, false otherwise
     */
    [[nodiscard]] constexpr bool isValid(int r, int q) const noexcept {
        return validityMask[getValidityMaskCoord(r, q)];
    }

    [[nodiscard]] std::vector<Point *> getNeighbors(const Point &point);

    [[nodiscard]] constexpr std::span<const std::pair<int, int>, 6> getNeighborOffsets() const noexcept {
        return neighborOffsets;
    }

    [[nodiscard]] constexpr std::pair<int, int> convertOffsetToAxial(std::pair<int, int> offsetCoords) const noexcept {
        int i = offsetCoords.first;
        int j = offsetCoords.second;
        int q = j;
        int r = q / 2 + i;
        return {q, r};
    }

    [[nodiscard]] constexpr std::pair<int, int> convertAxialToOffset(std::pair<int, int> coords) const noexcept {
        int q = coords.first;
        int r = coords.second;
        int j = q;
        int i = r - q / 2;
        return {i, j};
    };

    [[nodiscard]] double euclideanDistance(const Point &lhs, const Point &rhs) const;

    ~AxialRectangularMap() = default;

private:
    int width;
    int height;
    int maxCoordQ;
    int maxCoordR;
    std::pair<int, int> storageDims;

    /**
     * Array-of-structs storage: each element contains all data for one cell.
     * Indexed linearly with padding for SIMD operations.
     * See: https://www.redblobgames.com/grids/hexagons/#map-storage
     */
    std::vector<CellData> cells{};

    /**
     * Validity mask indicating which cells are valid (within hexagonal bounds).
     * Uses uint8_t instead of bool for better performance and cache locality.
     */
    std::vector<uint8_t> validityMask{};

    // TODO: This odd vs even distinction shouldn't be necessary.
    // The current implementation uses Offset coordinates. If Axial coordinates were used
    // the neighbor offsets would be the same for all cells.
    // The Offset coordinates were used here because it leads to no waste in storage.
    // That however complicates the processing. 
    constexpr static std::array<std::array<int, 2>, 6> axialDirectionVectorsOdd{
            {
                    {1, 0},
                    {-1, 0},
                    {0, 1},
                    {0, -1},
                    {-1, 1},
                    {-1, -1}
            }
    };

    constexpr static std::array<std::array<int, 2>, 6> axialDirectionVectorsEven{
            {
                    {1, 0},
                    {-1, 0},
                    {0, 1},
                    {0, -1},
                    {1, -1},
                    {1, 1}
            }};

    // TODO: is this used? This was probably intended for axial coordinates, which are
    // not used in this class in the end.
    constexpr static std::array<std::pair<int, int>, 6> neighborOffsets{
            {
                    {-1, -1},
                    {-1, 0},
                    {0,  1},
                    {1,  1},
                    {0,  -1},
                    {1,  0}
            }
    };

    void initializeStorageSize();

    void initializeStoragePoints();

    [[nodiscard]] bool areCoordsOutOfBounds(int q, int r) const;
};

// Include template implementation
#include "AxialRectangularMap.tpp"

