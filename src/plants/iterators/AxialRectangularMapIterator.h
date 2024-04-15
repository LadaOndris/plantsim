//
// Created by lada on 4/14/24.
//

#ifndef PLANTSIM_AXIALRECTANGULARMAPITERATOR_H
#define PLANTSIM_AXIALRECTANGULARMAPITERATOR_H



#include "Iterator.h"
#include <vector>

class AxialRectangularMap;

class AxialRectangularMapIterator : public Iterator {
public:
    AxialRectangularMapIterator(const std::vector<std::vector<std::shared_ptr<Point>>>& storage,
                                std::size_t widthStorageOffset,
                                std::size_t row,
                                std::size_t col);

    std::shared_ptr<Point> operator*() const override;
    Iterator& operator++() override;
    bool operator!=(const Iterator& other) const override;

private:
    void skipUnusedCells();

    const std::vector<std::vector<std::shared_ptr<Point>>>& storage;
    std::size_t widthStorageOffset;
    std::size_t row;
    std::size_t col;
};


#endif //PLANTSIM_AXIALRECTANGULARMAPITERATOR_H
