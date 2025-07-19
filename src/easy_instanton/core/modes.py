import pickle
from abc import ABC

import numpy as np
from numpy.linalg import norm
from numpy.typing import NDArray
from src.easy_instanton.core.opt.hessian import bofill
from src.easy_instanton.core.pes.abc import PES


class Data(ABC):
    """
    todo: shape
    """
    def __init__(self, x: NDArray, n_zero: int = 0, pes: PES | None = None):
        assert n_zero in (0, 3, 5, 6)
        self.x = x
        self.n_zero = n_zero
        self.pes = pes
        self.pot, self.grad, self.hess = pes.all(x)

    def update(self, x: NDArray, calc_hess: bool = False) -> None:
        x_old, grad_old, self.x = self.x, self.grad, x
        if calc_hess:
            self.pot, self.grad, self.hess = self.pes.all(self.x)
        else:
            self.pot, self.grad = self.pes.both(self.x)
            self.hess = bofill(self.hess, (self.x - x_old).ravel(), (self.grad - grad_old).ravel())

    def move(self, dx: NDArray) -> None:
        self.update(self.x + dx)

    def __str__(self):
        """used for optimization only"""
        return f'V = {self.pot:.5f}, |G| = {norm(self.grad):.5e}'

    def output(self, prefix: str) -> None:
        # todo: save traj, xyz
        comment = f'V = {self.pot:.18f}' if self.pot is not None else ''
        np.savetxt(prefix+'.txt', self.x, fmt='%15.8f', header=comment)

    def final_output(self, prefix: str) -> None:
        self.hess = self.pes.hessian(self.x)
        self.output(prefix)
        # todo: print freq
        with open(prefix+'.pkl', 'wb') as f:
            pickle.dump(self, f)


class Minimum(Data):
    order = 0
