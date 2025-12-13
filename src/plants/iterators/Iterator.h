#pragma once

#include <memory>
#include "plants/Point.h"

class Iterator {
public:
    virtual ~Iterator() = default;
    virtual std::shared_ptr<Point> operator*() const = 0;
    virtual Iterator& operator++() = 0;
    virtual bool operator!=(const Iterator& other) const = 0;
};


