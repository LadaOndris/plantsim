//
// Created by lada on 1/31/22.
//

#ifndef PLANTSIM_AXIALRECTANGULARMAP_H
#define PLANTSIM_AXIALRECTANGULARMAP_H

#include <vector>
#include <array>
#include "Map.h"

class AxialRectangularMap {

public:
    /**
     * Initializes the storage of the map.
     * It uses the axial coordiantes (q, r).
     *
     * @param width The maximmum q coordinate.
     * @param height The maximum r coordinate.
     */
    explicit AxialRectangularMap(int width, int height);

    [[nodiscard]] int getWidth() const;

    [[nodiscard]] int getHeight() const;

    [[nodiscard]] std::pair<int, int> getMaxCoords() const;

    [[nodiscard]] Point *getPoint(int x, int y);

    [[nodiscard]] std::vector<Point *> &getPoints();

    [[nodiscard]] std::vector<Point *> getNeighbors(const Point &point);

    [[nodiscard]] const std::vector<std::pair<int, int>> &getNeighborOffsets() const;

    [[nodiscard]] double euclideanDistance(const Point &lhs, const Point &rhs) const;

    ~AxialRectangularMap() = default;


private:
    int width;
    int height;
    int maxCoordQ;
    int maxCoordR;
    std::pair<int, int> storageDims;
    /**
     * 2D array is used as the storage representation.
     * See the following for more information:
     * https://www.redblobgames.com/grids/hexagons/#map-storagehttps://www.redblobgames.com/grids/hexagons/#map-storage
     */
    std::vector<Point> storage{};
    /**
     * Contains the flat representation of the storage without
     * the unused/invalid points.
     */
    std::vector<Point *> validPoints{};
    /**
     * Some of the cells in storage are unused due to the hexagonal nature
     * and rectangle shape. Thus, each line is padded has a padding
     * in front of it or at the back. The padding of each line is
     * expressed by this variable.
     */

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
            {-1, -1}, {-1, 0}, {0, 1}, {1, 1}, {1, 0}, {0, -1}
    };


    void initializeStorageSize();

    void initializeStoragePoints();

    [[nodiscard]] bool areCoordsOutOfBounds(int q, int r) const;

};


#endif //PLANTSIM_AXIALRECTANGULARMAP_H
