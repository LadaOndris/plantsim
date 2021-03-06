//
// Created by lada on 1/31/22.
//

#ifndef PLANTSIM_AXIALRECTANGULARMAP_H
#define PLANTSIM_AXIALRECTANGULARMAP_H

#include "Map.h"

class AxialRectangularMap : public Map {

public:
    AxialRectangularMap(std::size_t width, std::size_t height);

    size_t getWidth() const override;

    size_t getHeight() const override;

    std::shared_ptr<Point> getPoint(int x, int y) override;

    std::vector<std::shared_ptr<Point>> getNeighbors(std::shared_ptr<Point> point) override;

    double euclideanDistance(std::shared_ptr<Point> lhs, std::shared_ptr<Point> rhs) override;

    ~AxialRectangularMap() override = default;

private:
    std::size_t width;
    std::size_t height;
    /**
     * 2D array is used as the storage representation.
     * See the following for more information:
     * https://www.redblobgames.com/grids/hexagons/#map-storagehttps://www.redblobgames.com/grids/hexagons/#map-storage
     */
    std::vector<std::vector<std::shared_ptr<Point>>> storage;
    /**
     * Some of the cells in storage are unused due to the hexagonal nature
     * and rectangle shape. Thus, each line is padded has a padding
     * in front of it or at the back. The padding of each line is
     * expressed by this variable.
     */
    std::size_t widthStorageOffset;

    void initializeStorageSize();
    void initializeStoragePoints();
    bool areCoordsOutOfBounds(int q, int r) const;
};


#endif //PLANTSIM_AXIALRECTANGULARMAP_H
