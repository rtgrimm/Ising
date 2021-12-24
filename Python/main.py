import Ising
import matplotlib.pyplot as plt
from Ising import Index2D, Parameters
from Python.Ising import Ensemble
from Python.wrap import VecToMat
import  numpy as np


def test_1d_result():
    for J, B in [
        (0.0, 1.0),
        (1.0, 0.0),
        (1.0, 0.5),
        (-1.0, 0.5),
        (0.5, -1.0),
        (-0.5, -1.0),
    ]:
        plt.figure()
        free_energy_list = []
        exact_energy_list = []
        param_list = []

        #exact_free_energy = -J - (1/beta) * np.log(np.cosh(beta * B) + np.sqrt(np.sinh(B*beta)**2 + np.exp(-4*J*beta)))

        L = 50

        for beta in np.linspace(0.1, 3.0):
            print(L)
            size = Index2D(L, 1)


            lattice = Ising.Lattice(size, Parameters(-J, -B, beta))

            metropolis = Ising.Metropolis(lattice, 100)

            metropolis.rand_init()



            metropolis.step(100000)



            steps = 10000

            ensemble = Ensemble(metropolis, lattice)
            ensemble.sample(100000, 100)

            exact_free_energy = -(1/beta) * np.log(
                np.exp(beta * J) * np.cosh(beta * B)
                + np.sqrt(np.exp(2 * beta * J) * (np.sinh(beta * B))**2 + np.exp(-2*beta*J))
            )

            Z = ensemble.partition_function()
            free_energy = -(1 / (beta * L)) * np.log(Z)
            print(f"Z:{Z}; Free Energy: {free_energy}; Exact{exact_free_energy};")

            param_list.append(beta)
            free_energy_list.append(free_energy)
            exact_energy_list.append(exact_free_energy)



        plt.plot(param_list, free_energy_list)
        plt.plot(param_list, exact_energy_list)


    plt.show()



def display_lattice(dim, lattice):
    plt.clf()
    plt.title(lattice.parameters.beta)
    output = VecToMat(lattice.data())
    grid = output.reshape((dim, dim))

    plt.imshow(grid)
    plt.draw()
    plt.pause(1e-6)


def main():
    heat_cap_list = []
    T_list = []
    magnetization_list = []

    for T in np.linspace(0.1, 5.0, 200):
        beta = 1/T
        print(T)
        J = -1.0
        B = 0.0

        xdim = 12
        ydim = 12

        size = Index2D(xdim, ydim)

        lattice = Ising.Lattice(size, Parameters(J, B, beta))
        metropolis = Ising.Metropolis(lattice, 100)
        ensemble = Ising.Ensemble(metropolis, lattice)

        metropolis.rand_init()

        metropolis.step(10000)
        display_lattice(xdim, lattice)

        #plt.show()

        ensemble.sample(1000000, 10)

        magnetization = ensemble.magnetization()

        heat_cap = (1/T**2)*(ensemble.energy_variance() - ensemble.energy_mean()**2)
        heat_cap_list.append(heat_cap)
        magnetization_list.append(magnetization)
        T_list.append(T)

    plt.ioff()

    plt.figure()
    plt.scatter(T_list, heat_cap_list)

    plt.figure()
    plt.scatter(T_list, np.abs(magnetization_list))

    plt.show()



if __name__ == '__main__':
    main()