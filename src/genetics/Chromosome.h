#pragma once

#include "Gene.h"
#include <vector>
#include <stdexcept>
#include <iostream>


template<typename TGene>
class Chromosome {
public:
    explicit Chromosome(int length);

    std::vector<Gene<TGene>> &getGenes();

    int getLength() const;

    Gene<TGene> getGene(int index) const;

private:
    int length;
    std::vector<Gene<TGene>> genes;

    //template<typename T>
    //friend std::ostream& operator<<(std::ostream&, const Chromosome<T>&);
};

template<typename TGene>
std::ostream &operator<<(std::ostream &s, Chromosome<TGene> &chromosome) {
    s << "[";
    for (auto &gene : chromosome.getGenes()) {
        s << gene.getValue();
    }
    return s << "]" << std::endl;
}

template<typename TGene>
Chromosome<TGene>::Chromosome(int length)
        : genes(length), length(length) {
    //if (length < 2) {
    //    throw std::invalid_argument("Invalid length in Chromosome. Should be at least 2. Found: " + std::to_string(length));
    //}
    length = length;
}

template<typename TGene>
std::vector<Gene<TGene>> &Chromosome<TGene>::getGenes() {
    return genes;
}


template<typename TGene>
Gene<TGene> Chromosome<TGene>::getGene(int index) const {
    if (index < 0 || index >= length) {
        throw std::out_of_range("index");
    }
    return genes[index];
}

template<typename TGene>
int Chromosome<TGene>::getLength() const {
    return length;
}


