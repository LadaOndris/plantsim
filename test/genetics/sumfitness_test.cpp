
#include <gtest/gtest.h>
#include "genetics/SumFitness.h"
#include "genetics/Population.h"
#include "genetics/Chromosome.h"
#include "genetics/Gene.h"
#include "genetics/Individual.h"
#include "../dummies/TestIndividual.h"


TEST(SumFitness, CorrectMaxFitness) {
    // Initialize population
    std::unique_ptr<SumFitness<TestIndividual<int>>> fitness(std::make_unique<SumFitness<TestIndividual<int>>>());
    int populationSize = 10;
    int chromosomeLength = 20;
    Population<TestIndividual<int>> population(fitness, populationSize, chromosomeLength);

    // Set all genes to 1
    for (TestIndividual<int> &ind : population.getIndividuals()) {
        Chromosome<int> chromosome = ind.getChromosome();
        for (Gene<int> &gene : chromosome.getGenes()) {
            gene.setValue(1);
        }
    }

    // Compute fitness of each individual in population
    population.evaluate();

    // Check the maximum fitness is correct
    double maxFitness = population.getMaxFitness();
    ASSERT_EQ(maxFitness, chromosomeLength);
}


TEST(SumFitness, CorrectSumFitness) {

}

