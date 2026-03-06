from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import IntEnum
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pyrinst.geometries import Geometry


class Task(IntEnum):
    SP = 0  # Single Point
    GRAD = 1  # Gradient
    FREQ = 2  # Frequency


class Potential(ABC):
    """
    Base class upon which all other potentials should be based. ABC is an abstract class that does not allow
    instantiation. By default, hessian is computed by finite differences.
    """

    dx: float = 1e-4

    @abstractmethod
    def __call__(self, x: NDArray, task: Task = Task.GRAD) -> tuple[float, NDArray | None, NDArray | None]: ...

    def compute(self, geom: "Geometry", task: Task = Task.GRAD) -> None:
        """
        Compute the potential-energy surface properties.

        The attributes `energy`, `grad`, and optionally `hess` of the input `geometry` object will be updated in-place
        upon successful completion.

        Parameters
        ----------
        geom : Geometry
            The data container holding atomic coordinates and symbols.
        task : Task, optional
            The task to perform, by default Task.GRAD.
        """
        if geom.x.ndim == 3:
            func: Callable[[NDArray], tuple[float, NDArray | None, NDArray | None]] = partial(self, task=task)
            res = [np.array([res[i] for res in tuple(map(func, geom.x))]) for i in range(3)]
        else:
            res = self(geom.x, task)
        geom.V = res[0]
        geom.G = res[1]
        geom.H = res[2]


#    def compute_hess_from_grad(self, geom: "Geometry") -> None:
#        dim = geom.x.size
#        geom_tmp = Geometry(geom.x, geom.symbols)
#        res = np.empty((dim, dim), float)
#        dx = np.zeros_like(geom.x)
#        for i in range(dim):
#            dx.flat[i] = self.dx
#            geom_tmp.x = geom.x + dx
#            self.compute(geom_tmp)
#            f1 = geom_tmp.G.ravel()
#            geom_tmp.x = geom.x - dx
#            self.compute(geom_tmp)
#            f2 = geom_tmp.G.ravel()
#            res[i] = (f1 - f2) * 0.5 / self.dx
#            dx.flat[i] = 0
#        geom.H = 0.5 * (res + res.T)
