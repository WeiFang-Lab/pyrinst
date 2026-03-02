"""Routines for converting between coordinates"""

import numpy as np


def mass_weight(hess, mass, dim: int = 3):
    """
    Parameters
    ----------
    hess : NDArray
        Hessian at geometry x and has shape (x.size, x.size)
    mass : NDArray
        has shape (N_atom,)
    dim : int
        number of dimensions of each atom

    Returns
    -------
    NDArray
        mass-weighted Hessian
    """
    size = len(hess)
    shape = (size // (len(mass) * dim), len(mass), dim)
    sqrt_m = np.sqrt(mass)[None, ..., None]
    return (hess.reshape(2 * shape) / np.outer(sqrt_m, sqrt_m).reshape(2 * sqrt_m.shape)).reshape(size, size)


def is_linear(x) -> bool:
    if x.ndim == 2:
        x = x[None, ...]
    if x.shape[1] < 3:
        return True
    for xi in x:
        dx = xi[1:] - xi[0]
        if not all(np.isclose(np.linalg.norm(np.cross(dx[0], dx[i])), 0) for i in range(1, dx.shape[0])):
            return False
    return True
