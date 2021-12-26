import Ising
import matplotlib.pyplot as plt
from Ising import Index2D, Parameters
from Python.Ising import Ensemble
from Python.style import set_style, pal
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


def phase_transition():
    heat_cap_list = []
    T_list = []
    magnetization_list = []
    mag_sus_list = []

    k_b = 1

    for T in np.linspace(0.1, 5.0, 1000):
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

        magnetization = ensemble.magnetization_mean()

        mag_sus = (1/(k_b * T)) * (ensemble.magnetization_variance() - np.power(ensemble.magnetization_mean(), 2))


        heat_cap = (1/(k_b * T**2))*(ensemble.energy_variance() - np.power(ensemble.energy_mean(), 2))

        heat_cap_list.append(heat_cap)
        magnetization_list.append(magnetization)
        mag_sus_list.append(mag_sus)
        T_list.append(T)

    plt.ioff()

    colors = pal("muted", 3)

    plt.figure(figsize=(12, 12))

    plt.xlabel("$T$")
    plt.ylabel("$C_v$")
    plt.scatter(T_list, heat_cap_list, c=next(colors))
    plt.tight_layout()
    plt.savefig("heat_cap.pdf")

    plt.figure(figsize=(12, 12))
    plt.xlabel("$T$")
    plt.ylabel("$\\langle M \\rangle$")
    plt.scatter(T_list, np.abs(magnetization_list), c=next(colors))
    plt.tight_layout()
    plt.savefig("mag.pdf")

    plt.figure(figsize=(12, 12))
    plt.xlabel("$T$")
    plt.ylabel("$\chi_{mag}$")
    plt.scatter(T_list, mag_sus_list, c=next(colors))
    plt.tight_layout()
    plt.savefig("mag_sus.pdf")


    plt.show()


def quench():
    beta = 1.0
    J = -10.0
    B = 0.0

    dim = 500
    size = Index2D(dim, dim)

    lattice = Ising.Lattice(size, Parameters(J, B, beta))
    metropolis = Ising.Metropolis(lattice, 123)
    metropolis.rand_init()

    metropolis.step(2000000)



    plt.figure(figsize=(16, 16))
    output = VecToMat(lattice.data())
    grid = output.reshape((dim, dim))

    plt.xlabel("Site $X$")
    plt.ylabel("Site $Y$")
    plt.imshow(grid)
    plt.tight_layout()
    plt.savefig("grid.pdf")
    plt.show()

def EQ():
    beta = 1.0
    J = -10.0
    B = 0.0

    dim = 128
    size = Index2D(dim, dim)

    lattice = Ising.Lattice(size, Parameters(J, B, beta))
    metropolis = Ising.Metropolis(lattice, 123)
    metropolis.rand_init()

    energy_list = []
    steps = range(0, 10000)

    for i in steps:
        print(i / np.max(steps) * 100)
        energy_list.append(lattice.total_energy() / (dim * dim))
        metropolis.step(10000)

    plt.figure(figsize=(16, 16))
    plt.xlabel("Iteration")
    plt.ylabel("Energy per Site")
    plt.plot(np.array(steps) * 10000, energy_list)
    plt.tight_layout()
    plt.savefig("eq.pdf")
    plt.show()

if __name__ == '__main__':
    set_style()
    EQ()