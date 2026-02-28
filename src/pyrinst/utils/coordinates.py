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
