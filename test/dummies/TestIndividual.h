#pragma once

#include "genetics/SumFitness.h"
#include "genetics/Population.h"
#include "genetics/Chromosome.h"
#include "genetics/Gene.h"
#include "genetics/Individual.h"
#include "Random.h"

template<typename TGene>
class TestIndividual : public Individual<TGene> {
public:
    TestIndividual<TGene>() : Individual<TGene>()  {
    }

    explicit TestIndividual(int chromosomeLength) : Individual<TGene>(chromosomeLength) {
    }

    std::shared_ptr<Individual<TGene>> crossover(Individual<TGene> &other) override {
        int genesCount = this->chromosome.getLength();
        auto child = std::make_shared<TestIndividual<TGene>>(genesCount);

        auto &gen = Random::getGenerator();
        std::uniform_int_distribution<int> distribution(0, genesCount - 1); // Define the range
        int crossoverPoint = distribution(gen);
        //std::cout << crossoverPoint << std::endl;
        for (int i = 0; i < crossoverPoint; i++) {
            auto newGene = this->getChromosome().getGenes()[i].getValue();
            child->getChromosome().getGenes()[i].setValue(newGene);
        }
        for (int i = crossoverPoint; i < genesCount; i++) {
            auto newGene = other.getChromosome().getGenes()[i].getValue();
            child->getChromosome().getGenes()[i].setValue(newGene);
        }
        //std::cout << child->getChromosome();
        return child;
    }

    void mutate() override {
        int genesCount = this->chromosome.getLength();

        auto &gen = Random::getGenerator();
        std::uniform_int_distribution<int> distribution(0, genesCount - 1);

        int mutationPoint = distribution(gen);
        int previousValue = this->chromosome.getGenes()[mutationPoint].getValue();
        this->chromosome.getGenes()[mutationPoint].setValue(1 - previousValue);
    }
};


