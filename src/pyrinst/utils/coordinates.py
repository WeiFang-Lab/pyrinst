"""Routines for converting between coordinates"""

import numpy as np


def mass_weight(hess, mass=1, dim: int = 3):
    """
    Parameters
    ----------
    hess : NDArray
        Hessian at geometry x and has shape(x.size, x.size)
    mass : float | NDArray
        has shape x which broadcasts with x, e.g. (N_atom, 1)
    dim : int
        number of dimensions of each atom

    Returns
    -------
    NDArray
        mass-weighted Hessian
    """
    if np.isscalar(mass):
        return hess / mass
    assert dim in (1, 2, 3)
    size = len(hess)
    shape = (size // (len(mass) * dim), len(mass), dim)
    sqrt_m = np.sqrt(mass)[None, ..., None]
    return (hess.reshape(2 * shape) / np.outer(sqrt_m, sqrt_m).reshape(2 * sqrt_m.shape)).reshape(size, size)
