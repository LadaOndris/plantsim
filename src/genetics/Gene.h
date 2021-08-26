//
// Created by lada on 8/26/21.
//

#ifndef PLANTSIM_GENE_H
#define PLANTSIM_GENE_H


template<typename T>
class Gene {
public:
    explicit Gene(T value);
    friend bool operator==(const Gene<T> &lhs, const Gene<T> &rhs) {
        return lhs.value == rhs.value;
    }
private:
    T value;

};

template<typename T>
Gene<T>::Gene(T value)
        : value(value) {

}



#endif //PLANTSIM_GENE_H
