//
// Created by lada on 8/26/21.
//

#ifndef PLANTSIM_GENE_H
#define PLANTSIM_GENE_H


template<typename T>
class Gene {
public:
    Gene();

    explicit Gene(T value);

    void setValue(T value);

    friend bool operator==(const Gene<T> &lhs, const Gene<T> &rhs) {
        return lhs.value == rhs.value;
    }

private:
    T value;

};


template<typename T>
Gene<T>::Gene()
        : Gene<T>::Gene(0) {

}

template<typename T>
Gene<T>::Gene(T value)
        : value(value) {

}

template<typename T>
void Gene<T>::setValue(T other) {
    this->value = other;
}


#endif //PLANTSIM_GENE_H
