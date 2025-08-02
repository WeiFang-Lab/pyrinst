"""
Compute centre of mass and moments of inertia.

This module provides functions to calculate the center of mass and the moment of
inertia tensor for a collection of particles, such as a molecule or a system
of molecules. These calculations are fundamental in many areas of molecular
dynamics and physical chemistry.

N.B. For physically meaningful results, such as rotational constants, the
moment of inertia tensor should be computed with respect to the center of mass.
The `inertia` function handles this by default.
"""

__author__ = 'Jeremy O. Richardson'

import numpy as np
from numpy.typing import NDArray


def center_of_mass(x, mass: float | NDArray = 1):
    """Compute the center of mass of a system of particles.

    This function calculates the center of mass for a single system or a batch
    of systems.

    Parameters
    ----------
    x : numpy.ndarray
        An array of particle coordinates. The shape should be `(..., N, 3)`,
        where N is the number of particles. For example, `(N, 3)` for a single
        system or `(B, N, 3)` for a batch of B systems.
    mass : float or numpy.ndarray, optional
        The mass of each particle.
        If a float is given, all particles are assumed to have the same mass.
        If an ndarray is given, its shape should be broadcastable to the shape
        of `x` excluding the last dimension (e.g., `(..., N)`).
        Default is 1.0.

    Returns
    -------
    numpy.ndarray
        The coordinates of the center of mass. The shape will be `(..., 3)`.

    Examples
    --------
    >>> import numpy as np
    >>> # Coordinates for a water molecule (O, H, H) in Angstroms
    >>> coords = np.array([
    ...     [0.0000, 0.0000, 0.1173],
    ...     [0.0000, 0.7572, -0.4692],
    ...     [0.0000, -0.7572, -0.4692]
    ... ])
    >>> # Masses in atomic mass units (amu)
    >>> masses = np.array([15.999, 1.008, 1.008])
    >>> center_of_mass(coords, masses)
    array([0.        , 0.        , 0.00001664])
    """
    m = np.asarray(mass)
    return np.sum(m[..., None] * x, axis=-2) / np.sum(m)


def inertia(x, mass=1, com=True):
    """Compute the moment of inertia tensor.

    This function calculates the 3x3 moment of inertia tensor for a system of
    particles. By default, it first translates the system to its center of
    mass frame.

    Parameters
    ----------
    x : numpy.ndarray
        An array of particle coordinates. The shape must be `(..., N, 3)`,
        where N is the number of particles (e.g., `(N, 3)` or `(B, N, 3)`).
    mass : float or numpy.ndarray, optional
        The mass of each particle. Can be a float or an ndarray with a shape
        broadcastable to `(..., N)`. Default is 1.0.
    com : bool, optional
        If True (default), the system is first translated to its center of mass
        frame before the inertia tensor is calculated. This is necessary for
        most physical applications, like finding principal axes of rotation.

    Returns
    -------
    numpy.ndarray
        The 3x3 moment of inertia tensor. The shape will be `(..., 3, 3)`.

    Notes
    -----
    The moment of inertia tensor :math:`I` is defined as:

    .. math::

        I_{ij} = \\sum_k m_k [ (\\mathbf{r}_k \\cdot \\mathbf{r}_k) \\delta_{ij} - r_{ki} r_{kj} ]

    where :math:`m_k` is the mass of particle :math:`k` and :math:`\\mathbf{r}_k`
    is its position vector. For physically meaningful results, the coordinates
    :math:`\\mathbf{r}_k` should be relative to the center of mass.

    Examples
    --------
    >>> import numpy as np
    >>> # Coordinates for a water molecule (O, H, H) in Angstroms
    >>> coords = np.array([
    ...     [0.0000, 0.0000, 0.1173],
    ...     [0.0000, 0.7572, -0.4692],
    ...     [0.0000, -0.7572, -0.4692]
    ... ])
    >>> # Masses in atomic mass units (amu)
    >>> masses = np.array([15.999, 1.008, 1.008])
    >>> I = inertia(coords, masses)
    >>> # The principal moments of inertia are the eigenvalues of this tensor.
    >>> p_moments = np.linalg.eigvalsh(I)
    >>> print(np.round(I, 4))
    [[ 1.0253  0.      0.    ]
     [ 0.      0.0019  0.    ]
     [ 0.      0.      1.0272]]
    >>> print(np.round(p_moments, 4))
    [0.0019 1.0253 1.0272]
    """
    m = np.asarray(mass)  # convert to numpy arrays
    assert x.ndim == 2 or x.ndim == 3 and x.shape[-1] == 3
    if com:
        x = x - center_of_mass(x, m)  # avoid in-place modification
    diag = np.sum(m * np.sum(x**2, axis=-1))
    outer = np.sum(m[..., None] * x * x[..., None, :], axis=np.arange(x.ndim - 1))
    return np.eye(3) * diag - outer
