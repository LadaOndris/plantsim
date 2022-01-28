
#include <gtest/gtest.h>
#include "genetics/Chromosome.h"

// Demonstrate some basic assertions.
TEST(ChromosomeConstruction, NumberOfGenesEqualsLength) {
    int length = 10;
    Chromosome<int> chromosome(length);

    auto &genes = chromosome.getGenes();

    ASSERT_EQ(genes.size(), length);
}

void setGenesValue(Chromosome<int> &chromosome, int geneValue) {
    for (auto &gene : chromosome.getGenes()) {
        gene.setValue(geneValue);
    }
}

void assertGenesValue(Chromosome<int> &chromosome, int geneValue) {
    for (auto &gene : chromosome.getGenes()) {
        int value = gene.getValue();
        ASSERT_EQ(value, geneValue);
    }
}

TEST(ChromosomeModification, SetGeneValues) {
    int length = 1;
    Chromosome<int> chromosome(length);
    int geneValue = 100;
    setGenesValue(chromosome, geneValue);
    assertGenesValue(chromosome, geneValue);
}



