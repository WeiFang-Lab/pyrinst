"""build matrix P=I-sum(outer(pi,pi)) such that P.H.P.pi = 0
pi must be orthonormal.
For molecules, mass should have shape (N_atoms,1).
Explicitly pass mass only when projecting out a mass-weighted hessian. todo: doc
"""

__author__ = 'Jeremy O. Richardson'

import math
import numpy as np
from numpy.linalg import norm
from scipy import linalg
from pyrinst.utils import mechanics


def trans(x, mass=1):
    """Returns translational modes. Use argument mass to get mass-weighted
    translational modes"""
    dim = x.shape[-1]
    p = np.zeros((dim,) + x.shape)
    m = math.sqrt(x.size / dim)
    for i in range(dim):
        p[i, ..., i] = np.sqrt(mass) / m  # mass weighting
        p[i] /= norm(p[i])  # normalizing
    return p


def rot(x, mass=1):
    """Returns rotational modes. Use argument mass to get mass-weighted
    rotational modes"""
    assert x.shape[-1] == 3
    x -= mechanics.center_of_mass(x, mass)
    _, eig_vecs = linalg.eigh(mechanics.inertia(x, mass))
    x_rot = np.dot(x, eig_vecs)
    p = np.zeros((3,) + x.shape)
    for j in range(3):  # x, y, z
        p[..., j] = np.cross(x_rot * np.sqrt(mass)[..., None], eig_vecs[j], axisc=0)
    for j in range(3):
        p[j] /= norm(p[j])
    return p
