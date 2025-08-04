import numpy as np
from numpy.typing import NDArray
from pyrinst.core.pes.abc import PES, PESProxy


class Beads(PESProxy):
    def potential(self, x: NDArray) -> NDArray:
        return np.array(tuple(map(self._pes.potential, x)))

    def gradient(self, x: NDArray) -> NDArray:
        return np.array(tuple(map(self._pes.gradient, x)))

    def hessian(self, x: NDArray) -> NDArray:
        return np.array(tuple(map(self._pes.hessian, x)))

    def both(self, x: NDArray) -> list[NDArray]:
        return [np.array([res[i] for res in tuple(map(self._pes.both, x))]) for i in range(2)]

    def all(self, x: NDArray) -> list[NDArray]:
        return [np.array([res[i] for res in tuple(map(self._pes.all, x))]) for i in range(3)]


class Springs(PES):
    """half-ring"""
    def __init__(self, n: int, beta: float, mass: float | NDArray = 1):
        self.n = n
        self.beta = beta
        self.mass = mass
        self.hbar: float = self.units.hbar
        self.omega_n: float = n / (self.beta * self.hbar)

    def potential(self, x: NDArray) -> float:
        return self.omega_n ** 2 * np.sum(self.mass * np.sum(np.diff(x, axis=0) ** 2, 0))

    def gradient(self, x: NDArray) -> NDArray:
        res: NDArray = np.zeros_like(x)
        dx: NDArray = np.diff(x, axis=0)
        res[:-1] -= dx
        res[1:] += dx
        return res * 2 * self.mass * self.omega_n ** 2

    def hessian(self, x: NDArray) -> NDArray:  # todo: banded
        tmp: NDArray = (2 * np.ones_like(x[0]) * self.mass * self.omega_n**2).ravel()
        d: int = tmp.size
        res: NDArray = np.zeros((self.n//2*d, self.n//2*d))
        indices: NDArray = np.arange(len(res))
        res[indices[:-d], indices[:-d]] = tmp[indices[:-d] % d]
        res[indices[d:], indices[d:]] += tmp[indices[d:] % d]
        res[indices[:-d], indices[d:]] = res[indices[d:], indices[:-d]] = - tmp[indices[d:] % d]
        return res

    def hessian_full(self, x: NDArray) -> NDArray:
        tmp: NDArray = (np.ones_like(x[0]) * self.mass * self.omega_n**2).ravel()
        d: int = tmp.size
        res: NDArray = np.zeros((self.n*d, self.n*d))
        indices: NDArray = np.arange(len(res))
        res[indices, indices] = 2 * tmp[indices % d]
        res[indices, indices-d] = res[indices-d, indices] = - tmp[indices % d]
        return res
