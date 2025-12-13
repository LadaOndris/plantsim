#pragma once

#include <random>

class Random {
public:
    static std::mt19937 &getGenerator() {
        static std::random_device rd; // Obtain a random number from hardware
        static std::mt19937 gen(rd()); // Seed the generator
        return gen;
    }
};


