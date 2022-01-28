

#include <gtest/gtest.h>
#include <genetics/SumFitness.h>
#include "genetics/Chromosome.h"
#include "genetics/Gene.h"
#include "genetics/Individual.h"
#include "genetics/Population.h"
#include "../dummies/TestIndividual.h"

TEST(GeneticAlgorithm, ProblemConverges) {
    std::unique_ptr<IFitness<TestIndividual<int>>> fitness = std::make_unique<SumFitness<TestIndividual<int>>>();
    int populationSize = 10;
    int chromosomeLength = 20;
    Population<TestIndividual<int>> population(std::move(fitness), populationSize, chromosomeLength);
    population.evaluate();
    double initial_fitness = population.getMaxFitness();
    std::cout << "Max fitness: " << initial_fitness << std::endl;

    for (int i = 0; i < 20; i++) {
        population.select();
        population.crossover();
        population.mutate();
        population.evaluate();
        std::cout << "Max fitness: " << population.getMaxFitness() << std::endl;
    }
    double final_fitness = population.getMaxFitness();

    ASSERT_TRUE(final_fitness > initial_fitness);
}

