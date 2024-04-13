//
// Created by lada on 8/6/22.
//

#include "EntityChromosome.h"

EntityChromosome::EntityChromosome(unsigned NOptions, unsigned NHormones, unsigned NResources)
{
    hormoneProduction.reserve(NHormones * NOptions);
    hormoneDestruction.reserve(NHormones * NOptions);
    hormoneTransfer.reserve(6 * NHormones * NOptions);
    resourceTransfer.reserve(6 * NResources * NOptions);
    cellReplication.reserve(NOptions);
}

vector<HormoneOption> EntityChromosome::getHormoneProductionGenes() const {
    return hormoneProduction;
}

vector<HormoneOption> EntityChromosome::getHormoneDestructionGenes() const {
    return hormoneDestruction;
}

vector<HormoneOption> EntityChromosome::getHormoneTransferGenes() const {
    return hormoneTransfer;
}

vector<HormoneOption> EntityChromosome::getResourceTransferGenes() const {
    return resourceTransfer;
}

vector<HormoneOption> EntityChromosome::getCellReplicationGenes() const {
    return cellReplication;
}
