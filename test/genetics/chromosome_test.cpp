
#include <gtest/gtest.h>
#include "genetics/Chromosome.h"

// Demonstrate some basic assertions.
TEST(ChromosomeConstruction, NumberOfGenesEqualsLength) {
    int length = 10;
    Chromosome<int> chromosome(length);

    std::vector<Gene<int>> genes = chromosome.getGenes();

    ASSERT_EQ(genes.size(), length);
}