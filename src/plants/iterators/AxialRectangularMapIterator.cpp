

#include "AxialRectangularMapIterator.h"

AxialRectangularMapIterator::AxialRectangularMapIterator(
        const std::vector<std::vector<std::shared_ptr<Point>>> &storage,
        std::size_t widthStorageOffset,
        std::size_t row,
        std::size_t col)
        : storage(storage), widthStorageOffset(widthStorageOffset), row(row), col(col) {
    skipUnusedCells();
}

std::shared_ptr<Point> AxialRectangularMapIterator::operator*() const {
    return storage[row][col];
}

Iterator &AxialRectangularMapIterator::operator++() {
    ++col;
    int numColsInThisRow = storage[row].size();
    if (col >= numColsInThisRow) {
        ++row;
        col = 0;
    }
    skipUnusedCells();
    return *this;
}

bool AxialRectangularMapIterator::operator!=(const Iterator &other) const {
    const auto &rhs = dynamic_cast<const AxialRectangularMapIterator &>(other);
    return row != rhs.row || col != rhs.col;
}

void AxialRectangularMapIterator::skipUnusedCells() {
    while (row < storage.size() && col < storage[row].size() && storage[row][col] == nullptr) {
        ++col;
        if (col >= storage[row].size()) {
            ++row;
            col = 0;
        }
    }
}