import numpy as np
from numpy.typing import NDArray
from scipy import linalg


def bfgs(hess: NDArray, d: NDArray, dg: NDArray) -> NDArray:
    hy = linalg.blas.dgemv(1.0, hess, d)  # use scipy blas for efficiency
    dy = np.dot(d, dg)
    dhy = np.dot(d, hy)
    if dy <= 0 or dhy <= 0:
        return hess
    for i in range(len(hess)):  # memory efficient implementation without efficiency loss
        hess[i] += dg[i] * dg / dy - hy[i] * hy / dhy
    return hess


def powell(hess: NDArray, d: NDArray, dg: NDArray) -> NDArray:
    """update Cartesian Hessian using gradient; for TS searches
    d is change in position and dg change in gradient"""
    ddi = 1 / np.dot(d, d)
    y = dg - linalg.blas.dgemv(1.0, hess, d)  # use scipy blas for efficiency
    hess += ddi * (np.outer(y, d) + np.outer(d, y) - np.dot(y, d) * np.outer(d, d) * ddi)
    return hess


def bofill(hess: NDArray, d: NDArray, dg: NDArray) -> NDArray:
    """update Hessian according to Bofill, JCC 15, 1 (1994)  # todo: doc
    is equivalent to H += (1-phi)*MS + phi*Powell
    """
    xi = dg - linalg.blas.dgemv(1.0, hess, d)  # use scipy blas for efficiency
    d2 = np.dot(d, d)
    dxi = np.dot(d, xi)
    xi2 = np.dot(xi, xi)
    if d2 == 0 or xi2 <= np.finfo(float).eps**2 * max(np.dot(dg, dg), np.dot(hess @ d, hess @ d)):
        return hess
    # Algebraically equivalent mixing avoids division by d.xi (which may vanish).
    phi = np.clip(dxi**2 / (d2 * xi2), 0, 1)
    psb = (np.outer(xi, d) + np.outer(d, xi)) / d2 - dxi * np.outer(d, d) / d2**2
    return hess + dxi * np.outer(xi, xi) / (d2 * xi2) + (1 - phi) * psb
