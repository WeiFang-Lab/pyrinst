"""for A + BC --> AB + C reactions

either for collinear collisions in Jacobi coordinates
or in full space in Cartesian coordinates

"""

import bkmp
import numpy as np

from pyrinst.potentials import Potential, Task
from pyrinst.utils.elements import element_data
from pyrinst.utils.units import Mass


class Full(Potential):
    def __init__(self, *args: str):
        assert all(atom in ("H", "D") for atom in args)
        self.atoms = list(args)
        self.mass = element_data.get_masses(self.atoms) * Mass(1, "amu").get("au")
        self._check()

    def _check(self):
        assert len(self.atoms) == 3

    def __call__(self, x, task: Task = Task.GRAD):
        energy = self.potential(x)
        gradient = self.gradient(x) if task > Task.SP else None
        hessian = self.hessian(x) if task > Task.GRAD else None
        return energy, gradient, hessian

    def potential(self, x):
        return bkmp.cartpot(x.T)

    def both(self, x):
        pot, grad = bkmp.cartboth(x.T)
        return pot, grad.T

    def gradient(self, x):
        return self.both(x)[1]

    def hessian(self, x):
        dim = x.size
        res = np.empty((dim, dim), float)
        dx = np.zeros_like(x)
        for i in range(dim):
            dx.flat[i] = self.dx
            f1 = self.gradient(x + dx).ravel()
            f2 = self.gradient(x - dx).ravel()
            res[i] = (f1 - f2) * 0.5 / self.dx
            dx.flat[i] = 0
        return 0.5 * (res + res.T)


class H(Full):
    def _check(self):
        assert len(self.atoms) == 1

    def potential(self, x):
        return 0

    def both(self, x):
        return 0, np.zeros((1, 3))

    def hessian(self, x):
        return np.zeros((3, 3))


class H2(Full):
    def _check(self):
        assert len(self.atoms) == 2

    def potential(self, x):
        return super().potential(np.concatenate((x, 100 * np.ones((1, 3)))))

    def both(self, x):
        pot, grad = super().both(np.concatenate((x, 100 * np.ones((1, 3)))))
        return pot, grad[:2, :]


class CustomPES:
    def __new__(cls, *args, **kwargs):
        match len(args):
            case 1:
                return H(*args)
            case 2:
                return H2(*args)
            case 3:
                return Full(*args)
            case _:
                raise ValueError(f"Unknown input format: {args}")
