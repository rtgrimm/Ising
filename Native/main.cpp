

#include <iostream>
#include "Native/lattice.hpp"

int main() {
    Ising::Lattice lattice(Ising::Index2D {10, 10}, 123, Ising::Parameters {
        1.0,
        1.0,
        1.0
    });

    lattice.rand_init();

    auto x = lattice.flatten();

    for (int i = 0; i < 100; ++i) {
        lattice.step();
        std::cout << lattice.total_energy() << std::endl;
    }

    return 0;
}