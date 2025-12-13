#pragma once

#include <memory>
#include "genetics/Chromosome.h"
#include "Point.h"

class Cell : public Point {
public:
    explicit Cell(int x, int y);

    virtual ~Cell();

    [[nodiscard]] int getResources() const {
        return resources;
    }

    void setResources(int value) {
        this->resources = value;
    }

private:
    int resources;
};


