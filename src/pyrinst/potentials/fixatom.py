import numpy as np
from numpy.typing import NDArray

from pyrinst.utils.numderiv import grad_from_energy, hess_from_energy, hess_from_grad

from .base import Potential, Task


class FixAtom(Potential):
    def __init__(self, potential: Potential, x_fix: NDArray, dx=(None, None)):
        self.potential = potential
        self.n_fix = len(x_fix)
        self.x_fix = x_fix
        self.grad_dx, self.hess_dx = dx

    def __call__(self, x: NDArray, task: Task):
        x = np.r_[x, self.x_fix]
        if task == Task.FREQ:
            if self.hess_dx is None:
                res = list(self.potential(x, task))
            elif self.grad_dx is None:
                res = list(self.potential(x, task - 1))
                res[2] = hess_from_grad(x, lambda y: self.potential(y, task - 1)[1], self.hess_dx)
            else:
                res = list(self.potential(x, task - 2))
                res[1] = grad_from_energy(x, lambda y: self.potential(y, task - 2)[0], self.grad_dx)
                res[2] = hess_from_energy(x, lambda y: self.potential(y, task - 2)[1], self.hess_dx)
            res[1] = res[1][: -self.n_fix]
            res[2] = res[2][: -self.n_fix * 3, : -self.n_fix * 3]
        elif task == Task.GRAD:
            if self.grad_dx is None:
                res = list(self.potential(x, task))
            else:
                res = list(self.potential(x, task - 1))
                res[1] = grad_from_energy(x, lambda y: self.potential(y, task - 1)[0], self.grad_dx)
            res[1] = res[1][: -self.n_fix]
        else:
            res = self.potential(x, task)
        return tuple(res)
