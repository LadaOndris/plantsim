#include <iostream>
#include "Environment.h"

int main() {
    int width = 10;
    int height = 20;
    Environment env;
    WorldState state(width, height);

    std::cout << "Hello, World!" << std::endl;
    return 0;
}
