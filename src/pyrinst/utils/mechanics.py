"""Compute centre of mass and moments of inertia.
N.B. it is sometimes necessary to locate your molecule with its centre of mass at the origin.

For a usage example, run this file as a script. todo: doc
"""

__author__ = 'Jeremy O. Richardson'

import numpy as np
from numpy.typing import NDArray


def center_of_mass(x, mass: float | NDArray = 1):
    """Return centre of mass."""
    m = np.asarray(mass)
    return np.sum(m[..., None] * x, axis=-2) / np.sum(m)


def inertia(x, mass=1, com=True):
    """Return the moment of inertia tensor about the xyz axes centered on the origin.
    You MUST place your molecule at the CoM if you wish to calculate its rotational constants."""
    m = np.asarray(mass)  # convert to numpy arrays
    assert x.ndim == 2 or x.ndim == 3 and x.shape[-1] == 3
    if com:
        x = x - center_of_mass(x, m)  # avoid in-place modification
    diag = np.sum(m * np.sum(x**2, axis=-1))
    outer = np.sum(m[..., None] * x * x[..., None, :], axis=np.arange(x.ndim - 1))
    return np.eye(3) * diag - outer
