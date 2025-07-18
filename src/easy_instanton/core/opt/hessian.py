__author__ = 'Jeremy O. Richardson'

import numpy as np
from numpy.typing import NDArray
from scipy import linalg


def bofill(hess: NDArray, d: NDArray, dg: NDArray):
    """update Hessian according to Bofill, JCC 15, 1 (1994)  # todo: doc
    is equivalent to H += (1-phi)*MS + phi*Powell
    """
    xi = dg - linalg.blas.dgemv(1.0, hess, d)  # use scipy blas for efficiency
    d2 = np.dot(d, d)
    dxi = np.dot(d, xi)
    xi2 = np.dot(xi, xi)
    w = d/d2 - xi/dxi
    phi = 1 - dxi**2/(d2*xi2)
    return hess + np.outer(xi, xi)/dxi - phi*dxi*np.outer(w, w)
