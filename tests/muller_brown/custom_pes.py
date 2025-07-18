"""K. Muller and L.D. Brown, Theor. Chim. Acta 53, 75 (1979)"""

__author__ = 'Jeremy O. Richardson'

from collections.abc import Sequence
import numpy as np
import matplotlib.pyplot as plt
from src.easy_instanton.core.pes.abc import PES


class CustomPes(PES):
    def __init__(self):
        self.A = np.array([-200, -100, -170, 15])
        self.a = np.array([-1, -1, -6.5, 0.7])
        self.b = np.array([0, 0, 11, 0.6])
        self.c = np.array([-10, -10, -6.5, 0.7])
        self.x0 = np.array([1, 0, -0.5, -1])
        self.y0 = np.array([0, 0.5, 1.5, 1])

        self.min1 = [-0.55822363, 1.44172584]
        self.min2 = [-0.05001082, 0.4666941]
        self.min3 = [0.6234994, 0.02803776]
        self.TS1 = [-0.82200249, 0.62431232]
        self.TS2 = [0.21248636, 0.29298824]

        self.patches = []

        self.au2kcal = 627.509474063056
        # self.UNITS = units.hartAng()  todo
        # self.mass = np.array([1.00748, 1.00748]) / units.Mass(units.hartAng.mass).get('amu')
        self.atomlist = ['H', 'H']

    def potential(self, x: Sequence):
        x, y = x
        res = sum(self.A * np.exp(
            self.a * (x - self.x0) ** 2 + self.b * (x - self.x0) * (y - self.y0) + self.c * (y - self.y0) ** 2))
        return res / self.au2kcal

    def force(self, x):
        x, y = x
        force = np.zeros(2)
        tmp = self.A * np.exp(
            self.a * (x - self.x0) ** 2 + self.b * (x - self.x0) * (y - self.y0) + self.c * (y - self.y0) ** 2)
        force[0] = - sum(tmp * (self.a * 2 * (x - self.x0) + self.b * (y - self.y0)))
        force[1] = - sum(tmp * (self.c * 2 * (y - self.y0) + self.b * (x - self.x0)))
        return force / self.au2kcal

    def gradient(self, x):
        return -self.force(x)

    def hessian(self, x):
        x, y = x
        res = np.zeros((2, 2))
        tmp = self.A * np.exp(
            self.a * (x - self.x0) ** 2 + self.b * (x - self.x0) * (y - self.y0) + self.c * (y - self.y0) ** 2)
        res[0, 0] = sum(tmp * ((self.a * 2 * (x - self.x0) + self.b * (y - self.y0)) ** 2 + self.a * 2))
        res[0, 1] = res[1, 0] = sum(tmp * (
            (self.a * 2 * (x - self.x0) + self.b * (y - self.y0)) * (self.c * 2 * (y - self.y0) + self.b * (x - self.x0))
            + self.b) * 2)
        res[1, 1] = sum(tmp * ((self.c * 2 * (y - self.y0) + self.b * (x - self.x0)) ** 2 + self.c * 2))
        return res / self.au2kcal

    def plot(self, show_points=True):
        x, y = np.meshgrid(np.linspace(-1.5, 1.1, 200), np.linspace(-0.5, 2, 200))
        z = np.array([self.potential((xi, yi)) for xi, yi in zip(x.ravel(), y.ravel())]).reshape(x.shape)
        z_max = np.min(z) + 120 / self.au2kcal
        z[z > z_max] = np.nan
        plt.contourf(x, y, z, 61)
        plt.contour(x, y, z, 16, colors='k', linewidths=0.5, linestyles='solid')

        if show_points:
            for xt in (self.min1, self.min2, self.min3, self.TS1, self.TS2):
                plt.plot(xt[0], xt[1], 'ro')

        plt.show()


if __name__ == '__main__':
    MB = CustomPes()
    MB.plot()

