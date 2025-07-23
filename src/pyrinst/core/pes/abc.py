from numpy.typing import NDArray
from abc import ABC, abstractmethod


class PES(ABC):
    """
    Base class upon which all other PESs should be based. ABC is an abstract class that does not allow instantiation.
    By default, gradients and hessians are computed by finite differences and "both" and "all" functions simply call
    the separate routines.
    """
    mass: float | NDArray = 1.0
    dx: float = 1e-4

    @abstractmethod
    def potential(self, x: NDArray) -> float:  # this method should be overridden
        ...

    def gradient(self, x: NDArray) -> NDArray:
        ...  # todo: implement finite difference

    def hessian(self, x: NDArray) -> NDArray:
        ...  # todo: implement finite difference

    def force(self, x: NDArray) -> NDArray:
        return -self.gradient(x)

    def both(self, x: NDArray) -> tuple[float, NDArray]:
        return self.potential(x), self.gradient(x)

    def all(self, x: NDArray) -> tuple[float, NDArray, NDArray]:
        energy, grad = self.both(x)
        return energy, grad, self.hessian(x)
