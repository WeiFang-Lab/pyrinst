"""K. Muller and L.D. Brown, Theor. Chim. Acta 53, 75 (1979)"""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from pyrinst.potentials import Potential, Task
from pyrinst.utils.units import Length, Mass


class MullerBrown(Potential):
    def __init__(self):
        """
        Müller-Brown potential.
        Parameters are renamed as follows:
            height: A
            coeff_mat: [[a,   b/2],
                        [b/2, c]]
            x0: (x0, y0)
        """
        self.height = np.array([-200, -100, -170, 15])
        self.coeff_mat = np.zeros((4, 2, 2))
        self.coeff_mat[:, 0, 0] = np.array((-1, -1, -6.5, 0.7))
        self.coeff_mat[:, 0, 1] = np.array((0, 0, 11, 0.6))
        self.coeff_mat[:, 1, 1] = np.array((-10, -10, -6.5, 0.7))
        self.coeff_mat = 0.5 * (self.coeff_mat + self.coeff_mat.transpose(0, 2, 1))
        self.x0 = np.array(((1, 0), (0, 0.5), (-0.5, 1.5), (-1, 1)))

        self.min1 = [-0.55822363, 1.44172584]
        self.min2 = [-0.05001082, 0.4666941]
        self.min3 = [0.6234994, 0.02803776]
        self.TS1 = [-0.82200249, 0.62431232]
        self.TS2 = [0.21248636, 0.29298824]

        self.au2kcal = 627.509474063056
        self.masses = np.ones(2) * Mass(1.00748, "amu").get("au")

    def __call__(self, x, task: Task = Task.GRAD):
        energy = self.potential(x)
        gradient = self.gradient(x) if task > Task.SP else None
        hessian = self.hessian(x) if task > Task.GRAD else None
        return energy, gradient, hessian

    def potential(self, x: Sequence):
        dx = Length(x, "au").get("A")[..., 0] - self.x0
        return np.sum(self.height * np.exp(np.einsum("ij,ijk,ik->i", dx, self.coeff_mat, dx))) / self.au2kcal

    def gradient(self, x):
        dx = Length(x, "au").get("A")[..., 0] - self.x0
        tmp = self.height * np.exp(np.einsum("ij,ijk,ik->i", dx, self.coeff_mat, dx))
        res = np.sum(tmp[:, None] * 2 * np.einsum("ijk,ik->ij", self.coeff_mat, dx), axis=0)
        return res[..., None] / self.au2kcal / Length(1, "A").get("au")

    def hessian(self, x):
        dx = Length(x, "au").get("A")[..., 0] - self.x0
        tmp = self.height * np.exp(np.einsum("ij,ijk,ik->i", dx, self.coeff_mat, dx))
        tmp_vec = 2 * np.einsum("ijk,ik->ij", self.coeff_mat, dx)
        res = np.sum(tmp[:, None, None] * (np.einsum("ij,ik->ijk", tmp_vec, tmp_vec) + 2 * self.coeff_mat), axis=0)
        return res / self.au2kcal / Length(1, "A").get("au") ** 2

    def plot(self, show_points=True):
        x, y = np.meshgrid(np.linspace(-1.5, 1.1, 200), np.linspace(-0.5, 2, 200))
        z = np.array([self.potential((xi, yi)) for xi, yi in zip(x.ravel(), y.ravel(), strict=True)]).reshape(x.shape)
        z_max = np.min(z) + 120 / self.au2kcal
        z[z > z_max] = np.nan
        plt.contourf(x, y, z, 61)
        plt.contour(x, y, z, 16, colors="k", linewidths=0.5, linestyles="solid")

        if show_points:
            for xt in (self.min1, self.min2, self.min3, self.TS1, self.TS2):
                plt.plot(xt[0], xt[1], "ro")

        plt.show()


if __name__ == "__main__":
    MB = MullerBrown()
    MB.plot()
