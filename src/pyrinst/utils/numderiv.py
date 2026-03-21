"""Calculate numerical derivatives"""

import numpy as np


def grad_from_energy(x, func, dx):
    grad = np.empty_like(x)
    ei = np.zeros_like(x)
    for i in range(x.size):
        ei.flat[i] = dx
        grad.flat[i] = (func(x + ei) - func(x - ei)) / (2 * dx)
        ei.flat[i] = 0.0
    return grad


def hess_from_energy(x, func, dx):
    hess = np.empty((x.size, x.size))
    ei = np.zeros(x.shape)
    ej = np.zeros(x.shape)
    for i in range(x.size):
        ei.flat[i] = dx
        hess[i, i] = (func(x + ei) - 2 * func(x) + func(x - ei)) / dx**2
        for j in range(i):
            ej.flat[j] = dx
            hess[i, j] = (func(x + ei + ej) - 2 * func(x) + func(x - ei - ej)) / (2 * dx**2)
            hess[i, j] -= (hess[i, i] + hess[j, j]) / 2
            hess[j, i] = hess[i, j]
            ej.flat[j] = 0.0
        ei.flat[i] = 0.0
    return hess


def hess_from_grad(x, func, dx):
    ei = np.zeros_like(x)
    hess = np.empty((x.size, x.size))
    for i in range(x.size):
        ei.flat[i] = dx
        f1 = func(x + ei).ravel()
        f2 = func(x - ei).ravel()
        hess[i] = (f1 - f2) / (2 * dx)
        ei.flat[i] = 0.0
    return 0.5 * (hess + hess.T)
