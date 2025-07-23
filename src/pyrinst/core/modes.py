import logging
import pickle
from abc import ABC
from warnings import warn

import numpy as np
from numpy.linalg import norm
from numpy.typing import NDArray
from pyrinst.core.opt.hessian import bofill
from pyrinst.core.pes.abc import PES
from pyrinst.utils.coordinates import mass_weight

log = logging.getLogger(__name__)
logging.captureWarnings(True)


class Data(ABC):
    """
    todo: shape
    """
    order = None

    def __init__(self, x: NDArray, pes: PES, n_zero: int = 0):
        assert n_zero in (0, 3, 5, 6)
        self.x = x
        self.pes = pes
        self.mass = pes.mass
        self.n_zero = n_zero
        self.pot, self.grad, self.hess = pes.all(x)
        self.freq: NDArray | None = None

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
        hess_mw = mass_weight(self.hess, self.mass)
        evals = np.linalg.eigvalsh(hess_mw)  # todo: project
        self.freq = np.sqrt(abs(evals)) * np.sign(evals)
        freq_nonzero = self.freq[np.argpartition(np.abs(self.freq), self.n_zero)[self.n_zero:]]
        zpe = 0.5 * np.sum(freq_nonzero, where=freq_nonzero > 0)
        log.info(f'frequencies in cm-1:\n{self.freq}')  # todo: units, fmt
        log.info(f'H.O. ZPE = {zpe:.4f} cm-1')
        # check for negative eigenvalues
        if (n := sum(freq_nonzero < 0)) != self.order:
            warn(f"Wrong number of negative eigenvalues (expected {self.order}, got {n} instead)", RuntimeWarning)
        with open(prefix+'.pkl', 'wb') as f:
            pickle.dump(self, f)


class Minimum(Data):
    order = 0
