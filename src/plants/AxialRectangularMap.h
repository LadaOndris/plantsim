//
// Created by lada on 1/31/22.
//

#ifndef PLANTSIM_AXIALRECTANGULARMAP_H
#define PLANTSIM_AXIALRECTANGULARMAP_H

#include <vector>
#include <array>
#include "Map.h"

class AxialRectangularMap : public Map {

public:
    AxialRectangularMap(std::size_t width, std::size_t height);

    [[nodiscard]] size_t getWidth() const override;

    [[nodiscard]] size_t getHeight() const override;

    [[nodiscard]] Point *getPoint(int x, int y) override;

    [[nodiscard]] std::vector<Point> &getPoints() override;

    [[nodiscard]] std::vector<Point *> getNeighbors(const Point &point) override;

    [[nodiscard]] double euclideanDistance(const Point &lhs, const Point &rhs) const override;

    ~AxialRectangularMap() override = default;


private:
    std::size_t width;
    std::size_t height;
    /**
     * 2D array is used as the storage representation.
     * See the following for more information:
     * https://www.redblobgames.com/grids/hexagons/#map-storagehttps://www.redblobgames.com/grids/hexagons/#map-storage
     */
    std::vector<std::vector<Point>> storage;
    /**
     * Contains the flat representation of the storage without
     * the unused/invalid points.
     */
    std::vector<Point> validPoints;
    /**
     * Some of the cells in storage are unused due to the hexagonal nature
     * and rectangle shape. Thus, each line is padded has a padding
     * in front of it or at the back. The padding of each line is
     * expressed by this variable.
     */
    std::size_t widthStorageOffset;

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

    void initializeStorageSize();

    void initializeStoragePoints();

    [[nodiscard]] bool areCoordsOutOfBounds(int q, int r) const;

};


#endif //PLANTSIM_AXIALRECTANGULARMAP_H
