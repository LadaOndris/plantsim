

#include "genetics/SumFitness.h"
#include "genetics/Population.h"
#include "genetics/Chromosome.h"
#include "genetics/Gene.h"

TEST(SumFitness, PopulationEvaluatedCorrectly) {
    SumFitness fitness;
    int populationSize = 10;
    int chromosomeLength = 20;
    Population population(&fitness, populationSize, chromosomeLength);
    for (Individual &ind : population.getIndividuals()) {
        Chromosome &chromosome = ind.getChromosome();
        for (Gene &gene : chromosome.getGenes()) {
            gene.setValue(1);
        }
    }
    population.evaluate();
    double maxFitness = population.getMaxFitness();
    ASSERT_EQ(maxFitness, chromosomeLength);
}