//
// Created by lada on 1/29/22.
//

#ifndef PLANTSIM_TESTINDIVIDUALFACTORY_H
#define PLANTSIM_TESTINDIVIDUALFACTORY_H

#include "genetics/IndividualFactory.h"

template <typename TGene>
class TestIndividualFactory : public IndividualFactory<TGene> {
public:
    TestIndividualFactory() = default;

    std::shared_ptr<Individual<TGene>> create(int chromosomeLength) const override {
        return std::make_shared<TestIndividual<TGene>>(chromosomeLength);
    }
    ~TestIndividualFactory() override { };
};


#endif //PLANTSIM_TESTINDIVIDUALFACTORY_H
