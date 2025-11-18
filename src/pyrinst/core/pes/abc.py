import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod


class PES(ABC):
    """
    Base class upon which all other PESs should be based. ABC is an abstract class that does not allow instantiation.
    By default, gradients and hessians are computed by finite differences and "both" and "all" functions simply call
    the separate routines.
    """
    mass: float | NDArray = 1.0
    atoms: list[str] | None = None
    dx: float = 1e-4

    @abstractmethod
    def potential(self, x: NDArray) -> float:  # this method should be overridden
        ...

    def gradient(self, x: NDArray) -> NDArray:
        ...  # todo: implement finite difference

    def hessian(self, x: NDArray) -> NDArray:
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

    def force(self, x: NDArray) -> NDArray:
        return -self.gradient(x)

    def both(self, x: NDArray) -> tuple[float, NDArray]:
        return self.potential(x), self.gradient(x)

    def all(self, x: NDArray) -> tuple[float, NDArray, NDArray]:
        energy, grad = self.both(x)
        return energy, grad, self.hessian(x)


class PESProxy(ABC):
    """
    Abstract base class for a Potential Energy Surface (PES) proxy.

    This class is designed to encapsulate a PES object and transparently
    delegate attribute and method access to it.

    Parameters
    ----------
    pes : PES
        The Potential Energy Surface (PES) object to be encapsulated and proxied.

    Attributes
    ----------
    _pes : PES
        The internally encapsulated PES object.
    """
    def __init__(self, pes):
        self._pes = pes

    def __getattr__(self, name):
        return getattr(self._pes, name)

    def __setstate__(self, state):
        """Although this method behaves exactly the same as the default, it is required for pickling."""
        self.__dict__.update(state)
