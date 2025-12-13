#pragma once

#include "genetics/IndividualFactory.h"
#include "TestIndividual.h"

template <typename TGene>
class TestIndividualFactory : public IndividualFactory<TGene> {
public:
    TestIndividualFactory() = default;

    std::shared_ptr<Individual<TGene>> create(int chromosomeLength) const override {
        return std::make_shared<TestIndividual<TGene>>(chromosomeLength);
    }
    ~TestIndividualFactory() override { };
};


