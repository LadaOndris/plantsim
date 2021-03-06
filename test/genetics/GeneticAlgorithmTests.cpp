

#include <gtest/gtest.h>
#include <genetics/SumFitness.h>
#include "genetics/Chromosome.h"
#include "genetics/Gene.h"
#include "genetics/Individual.h"
#include "genetics/Population.h"
#include "../dummies/TestIndividual.h"
#include "../dummies/TestIndividualFactory.h"

TEST(GeneticAlgorithm, ProblemConverges) {
    std::unique_ptr<IFitness<int>> fitness = std::make_unique<SumFitness<int>>();
    int populationSize = 100;
    int chromosomeLength = 100;
    TestIndividualFactory<int> factory;
    Population<int> population(std::move(fitness), populationSize, factory, chromosomeLength);
    population.initialize();
    population.evaluate();
    double initial_fitness = population.getMaxFitness();
    std::cout << "Max fitness: " << initial_fitness << std::endl;

    for (int i = 0; i < 100; i++) {
        population.select();
        population.crossover();
        population.mutate();
        population.evaluate();
        std::cout << "Max fitness: " << population.getMaxFitness() << std::endl;
    }
    double final_fitness = population.getMaxFitness();

    ASSERT_TRUE(final_fitness > initial_fitness);
}

