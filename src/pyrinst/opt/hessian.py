import numpy as np
from numpy.typing import NDArray
from scipy import linalg


def bfgs(hess: NDArray, d: NDArray, dg: NDArray) -> NDArray:
    hy = linalg.blas.dgemv(1.0, hess, d)  # use scipy blas for efficiency
    dy = np.dot(d, dg)
    a1 = (1 + np.dot(d, hy) / dy) / dy
    for i in range(len(hess)):  # memory efficient implementation without efficiency loss
        hess[i] += a1 * d[i] * d - (d[i] * hy + hy[i] * d) / dy
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
    w = d / d2 - xi / dxi
    phi = 1 - dxi**2 / (d2 * xi2)
    return hess + np.outer(xi, xi) / dxi - phi * dxi * np.outer(w, w)
