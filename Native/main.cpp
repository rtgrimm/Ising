

#include <iostream>
#include "Native/lattice.hpp"

int main() {
    auto L = 16;



    Ising::Lattice lattice(Ising::Index2D {L, L});

    lattice.parameters.set_uniform_binary(-1.0, 0.0);

    Ising::Metropolis metropolis(&lattice, 123);

    metropolis.rand_init();

    metropolis.step(10000);

    Ising::Ensemble ensemble(&metropolis, &lattice);

    ensemble.sample(10000, 1000);

    //std::cout << -(1/params.beta) * std::log(ensemble.partition_function()) / (L);
    std::cout << ensemble.energy_mean();

    return 0;
}