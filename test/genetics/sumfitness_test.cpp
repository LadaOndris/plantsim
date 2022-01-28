
#include <gtest/gtest.h>
#include "genetics/SumFitness.h"
#include "genetics/Population.h"
#include "genetics/Chromosome.h"
#include "genetics/Gene.h"
#include "genetics/Individual.h"
#include "../dummies/TestIndividual.h"

TEST(SumFitness, CorrectMaxFitness) {
    // Initialize population
    int populationSize = 10;
    int chromosomeLength = 20;
    std::unique_ptr<IFitness<TestIndividual<int>>> fitness = std::make_unique<SumFitness<TestIndividual<int>>>();
    Population<TestIndividual<int>> population(std::move(fitness), populationSize, chromosomeLength);

    // Set all genes to 1
    for (auto &ind : population.getIndividuals()) {
        auto &chromosome = ind.getChromosome();
        for (auto &gene : chromosome.getGenes()) {
            gene.setValue(1);
        }
    }

    // Compute fitness of each individual in population
    population.evaluate();

    // Check the maximum fitness is correct
    double maxFitness = population.getMaxFitness();
    ASSERT_EQ(maxFitness, chromosomeLength);
}

