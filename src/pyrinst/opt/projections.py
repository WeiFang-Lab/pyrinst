"""build matrix P=I-sum(outer(pi,pi)) such that P.H.P.pi = 0
pi must be orthonormal.
For molecules, mass should have shape (N_atoms,1).
Explicitly pass mass only when projecting out a mass-weighted hessian. todo: doc
"""

import math

import numpy as np
from numpy.linalg import norm
from numpy.typing import NDArray
from scipy import linalg

from pyrinst.utils import mechanics


def trans(x, mass: float | NDArray = 1):
    """Returns translational modes. Use argument mass to get mass-weighted
    translational modes"""
    dim = x.shape[-1]
    p = np.zeros((dim,) + x.shape)
    m = math.sqrt(x.size / dim)
    for i in range(dim):
        p[i, ..., i] = np.sqrt(mass) / m  # mass weighting
        p[i] /= norm(p[i])  # normalizing
    return p


def rot(x, mass: float | NDArray = 1):
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


def centroid(x: NDArray, mass: float | NDArray = 1) -> NDArray:
    p = np.tile(np.eye(np.prod(x.shape[1:], dtype=int)), (1, len(x))).reshape(-1, *x.shape) * np.sqrt(mass[:, None])
    p /= np.linalg.norm(p[0].ravel())
    return p


def proj_eig(
    x: NDArray, hess: NDArray, n_zero: int, mass: float | NDArray = 1, constr_vecs: NDArray = None
) -> tuple[NDArray, NDArray]:
    match n_zero:
        case 0:
            p = np.array([]).reshape(0, *x.shape)
        case 3:
            p = trans(x, mass)
        case 5:
            p = np.concatenate((trans(x, mass), rot(x, mass)[1:]))
        case 6:
            p = np.concatenate((trans(x, mass), rot(x, mass)))
        case _:
            raise ValueError(f"n_zero must be 0, 3, 5, or 6, not {n_zero}")
    p.shape = (-1, x.size)
    if constr_vecs is not None:
        constr_vecs = constr_vecs.reshape(-1, x.size)
        constr_vecs /= np.linalg.norm(constr_vecs, axis=1, keepdims=True)
        for i in range(len(constr_vecs)):
            constr_vecs[i] -= p @ constr_vecs[i] @ p
            constr_vecs[i] /= norm(constr_vecs[i])
            p = np.concatenate((p, [constr_vecs[i]]))
    p_mat = np.identity(x.size) - np.einsum("ij,ik->jk", p, p)
    hess = p_mat @ hess @ p_mat
    eig_vals, eig_vecs = linalg.eigh(hess)
    n_zero = len(p)
    if n_zero == 0:
        return eig_vals, eig_vecs
    idx = np.argpartition(abs(eig_vals), n_zero)[:n_zero]
    return np.delete(eig_vals, idx), np.delete(eig_vecs, idx, axis=1)


def main():
    x = np.arange(3 * 3, dtype=float).reshape(3, 3)
    print("3D trans:", trans(x))
    print("3D rot:", rot(x))


if __name__ == "__main__":
    main()
