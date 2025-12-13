#pragma once

#include <vector>
#include "HormoneOption.h"

using namespace std;

class EntityChromosome {
public:
    EntityChromosome(unsigned NOptions, unsigned NHormones, unsigned NResources);

    vector<HormoneOption> getHormoneProductionGenes() const;
    vector<HormoneOption> getHormoneDestructionGenes() const;
    vector<HormoneOption> getHormoneTransferGenes() const;
    vector<HormoneOption> getResourceTransferGenes() const;
    vector<HormoneOption> getCellReplicationGenes() const;

private:
    vector<HormoneOption> hormoneProduction;
    vector<HormoneOption> hormoneDestruction;
    vector<HormoneOption> hormoneTransfer;
    vector<HormoneOption> resourceTransfer;
    vector<HormoneOption> cellReplication;

//    array<array<HormoneOption, NOptions>, NHormones> hormoneProduction;
//    array<array<HormoneOption, NOptions>, NHormones> hormoneDestruction;
//    array<array<HormoneOption, NOptions>, 6 * NHormones> hormoneTransfer;
//    array<array<HormoneOption, NOptions>, 6 * NResources> resourceTransfer;
//    array<HormoneOption, NOptions> cellReplication;
};


