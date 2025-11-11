//
// Created by lada on 1/31/22.
//

#ifndef PLANTSIM_AXIALRECTANGULARMAP_H
#define PLANTSIM_AXIALRECTANGULARMAP_H

#include <vector>
#include <array>
#include <cassert>
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

    [[nodiscard]] int getWidth() const;

    [[nodiscard]] int getHeight() const;

    [[nodiscard]] std::pair<int, int> getMaxCoords() const;

    [[nodiscard]] std::pair<int, int> getStorageDims() const;

    /**
     * Gets the linear storage index for the given axial coordinates.
     * @param r Row coordinate
     * @param q Column coordinate
     * @return Linear index into the storage array
     */
    [[nodiscard]] inline int getStorageCoord(int r, int q) const {
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
    [[nodiscard]] inline int getValidityMaskCoord(int r, int q) const {
        // Offset of one point on each side
        return (r + 1) * (storageDims.first + 2) + q + 1;
    }

    /**
     * Gets the cell data at the given linear storage index.
     * @param idx Linear storage index
     * @return Reference to the cell data
     */
    CellData &getCellAt(int idx) {
        return cells[idx];
    }

    const CellData &getCellAt(int idx) const {
        return cells[idx];
    }

    /**
     * Gets mutable reference to all cells (array-of-structs layout).
     * @return Reference to the cells vector
     */
    std::vector<CellData> &getCells() {
        return cells;
    }

    /**
     * Gets const reference to all cells (array-of-structs layout).
     * @return Const reference to the cells vector
     */
    const std::vector<CellData> &getCells() const {
        return cells;
    }

    std::vector<uint8_t> &getValidityMask() {
        return validityMask;
    }

    [[nodiscard]] const std::vector<uint8_t> &getValidityMask() const {
        return validityMask;
    }

    [[nodiscard]] Point *getPoint(int x, int y);

    [[nodiscard]] std::vector<Point *> &getPoints();

    [[nodiscard]] std::vector<Point *> getNeighbors(const Point &point);

    [[nodiscard]] const std::vector<std::pair<int, int>> &getNeighborOffsets() const;

    [[nodiscard]] inline std::pair<int, int> convertOffsetToAxial(std::pair<int, int> offsetCoords) const {
        int i = offsetCoords.first;
        int j = offsetCoords.second;
        int q = j;
        int r = q / 2 + i;
        return {q, r};
    }

    [[nodiscard]] inline std::pair<int, int> convertAxialToOffset(std::pair<int, int> coords) const {
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

    /**
     * Contains the flat representation of the storage without the unused/invalid points.
     */
    std::vector<Point *> validPoints{};

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

    const std::vector<std::pair<int, int>> neighborOffsets{
            {-1, -1},
            {-1, 0},
            {0,  1},
            {1,  1},
            {0,  -1},
            {1,  0}
    };

    void initializeStorageSize();

    void initializeStoragePoints();

    [[nodiscard]] bool areCoordsOutOfBounds(int q, int r) const;
};

// Include template implementation
#include "AxialRectangularMap.tpp"

#endif //PLANTSIM_AXIALRECTANGULARMAP_H
