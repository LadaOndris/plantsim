

#include <gtest/gtest.h>
#include "genetics/Chromosome.h"
#include "genetics/Gene.h"
#include "genetics/Individual.h"
#include "genetics/Population.h"

TEST(GeneticAlgorithm, ProblemConverges) {
    SumFitness fitness;
    Population population(&fitness);
    population.evaluate();
    double initial_fitness = population.getMaxFitness();
    std::cout << "Max fitness: " << initial_fitness << std::endl;

    for (int i = 0; i < 20; i++) {
        population.select();
        population.crossover();
        population.mutate();
        population.evaluate();
        std::cout << "Max fitness: " << populatoin.getMaxFitness() << std::endl;
    }
    double final_fitness = population.getMaxFitness();

    ASSERT_TRUE(final_fitness > initial_fitness);
}

