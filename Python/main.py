import Ising
import matplotlib.pyplot as plt
from Ising import Index2D, Parameters
from Python.wrap import VecToMat

dim = 100

size = Index2D(dim, dim)

lattice = Ising.Lattice(size, 124, Parameters(-1.0, 0.0, 100.0))

lattice.rand_init()

plt.ion()
plt.figure(figsize=(16, 9))

steps = 10000000

for i in range(0, steps):
    lattice.step()

    if i % 10000 == 0:
        plt.clf()
        output = VecToMat(lattice.flatten())
        grid = output.reshape((dim, dim))

        plt.title(f"Energy: {lattice.total_energy()}; Step% {i / steps * 100}")

        plt.imshow(grid)
        plt.draw()
        plt.pause(1e-6)

plt.ioff()
plt.show()

