#pragma once

#include <vector>
#include <cstdint>
#include <random>
#include <map>
#include <tuple>

namespace Ising {

    struct Index2D {
        int32_t x = 0;
        int32_t y = 0;

        Index2D(signed int x, signed int y) : x(x), y(y) {}

        Index2D(const Index2D&) = default;
        Index2D& operator= (const Index2D&) = default;
    };

    class Parameters {
    public:
        double beta = 1.0;

        void set_uniform_binary(double J, double B) {
            set_J(0, 1, -J);
            set_J(0, 0, J);
            set_B(0, -B);
            set_B(1, B);
        }

        void set_J(signed int i, signed int j, double J) {
            J_map[std::make_tuple(i, j)] = J;
            J_map[std::make_tuple(j, i)] = J;
        }

        double get_J(signed int i, signed int j)  {
            return J_map[std::make_tuple(i, j)];
        }
        
        double get_B(signed int i) {
            return B_map[i];
        }
        
        void set_B(signed int i, double B) {
            B_map[i] = B;
        }
        

    private:
        std::map<std::tuple<int32_t, int32_t>, double> J_map;
        std::map<int32_t, double> B_map;
    };


    class Lattice {
    public:
        Index2D size;
        Parameters parameters;


        std::vector<int32_t> data() {
            return sites;
        }

        explicit Lattice(Index2D size) : size(size)
        {
            sites.resize(size.x * size.y);
        }

        Lattice(const Lattice& other) = default;
        Lattice& operator=(const Lattice& other) = default;

        Lattice(Lattice&& other)  noexcept :
        size(other.size), parameters(other.parameters), sites(std::move(other.sites)) {}

        Lattice& operator= (Lattice&& other)  noexcept {
            size = other.size;
            parameters = other.parameters;
            sites = std::move(other.sites);

            return *this;
        }

        double boltzmann_factor() {
            return std::exp(-total_energy() * parameters.beta);
        }

        double magnetization() {
            double total = 0.0;

            for_all([&] (Index2D center) {
                total += read(center);
            });

            return total / (size.x * size.y);
        }

        double total_energy() {
            double total = 0.0;

            for_all([&] (Index2D center) {
                total += energy(center, read(center));
            });

            return total / 2;
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

        double energy(Index2D center, int32_t state) {
            auto interaction = [&] (auto dx, auto dy) {
                Index2D other_loc {center.x + dx, center.y + dy};
                auto other = read(other_loc);

                auto J = parameters.get_J(state, other);
                return J;
            };

            auto energy = parameters.get_B(state);

            energy += interaction(0, 1);
            energy += interaction(0, -1);
            energy += interaction(-1, 0);
            energy += interaction(1, 0);

            return energy;
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

    private:
        std::vector<int32_t> sites;

        size_t get_index(Index2D index) const {
            return index.x + index.y * size.x;
        }
    };

    class Metropolis {
    public:
        Metropolis(Lattice* lattice_, size_t seed, int state_count = 2) :
                        x_random(0, lattice_->size.x), y_random(0, lattice_->size.y),
                       state_random(0, state_count - 1), real_random(0.0, 1.0), lattice(lattice_), gen(seed) {}

        void rand_init() {
            lattice->for_all([&] (Index2D center) {
                lattice->write(center, gen_state());
            });
        }

        void fixed_init() {
            lattice->for_all([&] (Index2D center) {
                lattice->write(center, 1);
            });
        }

        void step(size_t steps = 1) {
            for (int step = 0; step < steps; ++step) {
                auto center = Index2D {x_random(gen), y_random(gen)};
                auto state = lattice->read(center);

                int32_t new_state = gen_state();

                auto delta_E = lattice->energy(center, new_state) - lattice->energy(center, state);

                if(delta_E <= 0) {
                    lattice->write(center, new_state);
                    continue;
                }

                auto r = real_random(gen);

                if(r <= std::exp(-lattice->parameters.beta * delta_E)) {
                    lattice->write(center, new_state);
                }
            }
        }

    private:
        Lattice* lattice;

        std::mt19937 gen;
        std::uniform_int_distribution<int32_t> x_random;
        std::uniform_int_distribution<int32_t> y_random;
        std::uniform_int_distribution<int32_t> state_random;
        std::uniform_real_distribution<double> real_random;

        int32_t gen_state() {
            return state_random(gen);
        }
    };

    class Ensemble {
    public:
        Ensemble(Metropolis* metropolis_, Lattice* lattice_)
        : metropolis(metropolis_), lattice(lattice_) {}

        void equilibrate(size_t steps, size_t max_steps, double delta_E_threshold) {
            double E_last = 0;
            
            for (auto i = 0; i < max_steps; ++i) {
                if(i % steps == 0 && i != 0) {
                    double current_E = lattice->total_energy();
                    double delta_E = current_E - E_last;

                    if (delta_E <= delta_E_threshold) {
                        break;
                    }
                }
            }
        }

        void sample(size_t steps, size_t sample_interval) {
            for (auto i = 0; i < steps; ++i) {
                if(i % sample_interval == 0 && i != 0) {
                    history.push_back(*lattice);
                }

                metropolis->step();
            }
        }

        double partition_function() {
            auto Z = 0.0;

            for(auto& config : history) {
                Z += config.boltzmann_factor();
            }

            return Z;
        }

        double magnetization_mean() {
            return expectation_value([&] (Lattice& config) {
                return config.magnetization();
            });
        }

        double magnetization_variance() {
            return expectation_value([&] (Lattice& config) {
                return std::pow(config.magnetization(), 2.0);
            });
        }

        double energy_mean() {
            return expectation_value([&] (Lattice& config) {
                return config.total_energy();
            });
        }

        double energy_variance() {
            return expectation_value([&] (Lattice& config) {
                return std::pow(config.total_energy(), 2.0);
            });
        }


        template<class F>
        double expectation_value(F callback) {
            auto ex = 0.0;

            for(auto& config : history) {
                ex += callback(config);
            }

            return ex / static_cast<double>(history.size());
        }

    private:
        std::vector<Lattice> history;

        Metropolis* metropolis;
        Lattice* lattice;
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