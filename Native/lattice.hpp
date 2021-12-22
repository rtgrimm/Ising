#pragma once

#include <vector>
#include <cstdint>
#include <random>

namespace Ising {

    struct Index2D {
        int32_t x = 0;
        int32_t y = 0;

        Index2D(signed int x, signed int y) : x(x), y(y) {}
    };

    struct Parameters {
        double J;
        double B;
        double Beta;

        Parameters(double j, double b, double beta) : J(j), B(b), Beta(beta) {}
    };

    class Lattice {
    public:
        std::vector<int32_t> sites;
        Index2D size;

        std::vector<std::vector<int32_t>> history;

        explicit Lattice(Index2D size, size_t seed, Parameters parameters) :
                size(size), gen(seed), parameters(parameters),
                x_random(0, size.x), y_random(0, size.y),
                state_random(0, 1), real_random(0.0, 1.0) {
            sites.resize(size.x * size.y);
        }

        double total_energy() {
            double total = 0.0;

            for_all([&] (Index2D center) {
                total += energy(center, read(center));
            });

            return total;
        }

        void step() {
            auto center = Index2D {x_random(gen), y_random(gen)};
            auto state = read(center);

            int32_t new_state = gen_state();

            auto delta_E = energy(center, new_state) - energy(center, state);

            if(delta_E <= 0) {
                write(center, new_state);
                return;
            }

            auto r = real_random(gen);

            if(r <= std::exp(-parameters.Beta * delta_E)) {
                write(center, new_state);
            }
        }

        void write(Index2D index, int32_t value) {
            auto x = index.x;
            auto y = index.y;

            if ((x >= 0 && x < size.x) && (y >= 0 && y < size.y)) {
                sites[get_index(index)] = value;
            }
        }

        int32_t read(Index2D index) {
            auto x = index.x;
            auto y = index.y;

            if ((x >= 0 && x < size.x) && (y >= 0 && y < size.y)) {
                return sites[get_index(index)];
            }

            return 0;
        }

        void rand_init() {
            for_all([&] (Index2D center) {
                write(center, gen_state());
            });
        }

        std::vector<int32_t> flatten() const {
            return sites;
        }

    private:
        Parameters parameters;
        std::mt19937 gen;
        std::uniform_int_distribution<int32_t> x_random;
        std::uniform_int_distribution<int32_t> y_random;
        std::uniform_int_distribution<int32_t> state_random;
        std::uniform_real_distribution<double> real_random;

        size_t get_index(Index2D index) const {
            return index.x + index.y * size.x;
        }

        int32_t gen_state() {
            if(state_random(gen) == 1) {
                return 1;
            } else {
                return -1;
            }
        }

        template<class F>
        void for_all(F func) {
            for (auto x = 0; x < size.x; x++) {
                for (auto y = 0; y < size.y; y++) {
                    auto center = Index2D{x, y};

                    func(center);
                }
            }
        }

        double energy(Index2D center, int32_t state) {
            auto interaction = [&] (auto dx, auto dy) {
                return double(read(Index2D{center.x + dx, center.y + dy}) * state) * parameters.J;
            };

            auto energy = double(parameters.B) * state;

            energy += interaction(0, 1);
            energy += interaction(0, -1);
            energy += interaction(-1, 0);
            energy += interaction(1, 0);

            return energy;
        }
    };

    struct VecData {
        int32_t* ptr;
        size_t size;
    };


    VecData vector_data(std::vector<int32_t> &target) {
        return {
                target.data(),
                target.size()
        };
    }

}