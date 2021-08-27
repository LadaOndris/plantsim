//`1
// Created by lada on 8/26/21.
//

#ifndef PLANTSIM_CHROMOSOME_H
#define PLANTSIM_CHROMOSOME_H

#include <boost/scoped_array.hpp>
#include "Gene.h"
#include <vector>
#include <stdexcept>


template<typename TGene>
class Chromosome {
public:
    explicit Chromosome(int length);

    std::vector<Gene<TGene>> getGenes() const;

    Gene<TGene> getGene(int index) const;

private:
    int length;
    std::vector<Gene<TGene>> genes;
};

template<typename TGene>
Chromosome<TGene>::Chromosome(int length)
        : genes(length) {
    if (length < 2) {
        throw std::invalid_argument("length");
    }
    length = length;
}

template<typename TGene>
std::vector<Gene<TGene>> Chromosome<TGene>::getGenes() const {
    return genes;
}

template<typename TGene>
Gene<TGene> Chromosome<TGene>::getGene(int index) const {
    if (index < 0 || index >= length) {
        throw std::out_of_range("index");
    }
    return genes[index];
}


#endif //PLANTSIM_CHROMOSOME_H
