import logging
import numpy as np
from numpy.linalg import norm
from scipy import linalg

log = logging.getLogger(__name__)


class NewtonRaphson:
    """Base class of all mode-following optimizers.
    The standard Newton-Raphson ignores argument order and just optimizes to any nearby stationary point.
    """
    def __init__(self, maxstep=None, project=None, update: bool = True):
        """
        verbosity -- controls messages
        """
        self.maxstep = maxstep
        self.project = project
        self.update = update

    def scale(self, h):
        if self.maxstep is not None:
            step = norm(h)
            if step > self.maxstep:
                h *= self.maxstep/step
        return h

    def iterate(self, data):
        """Take one iteration, including rescaling step"""
        # compute attempted step
        hess = data.hess.copy()
        h = - linalg.solve(hess, data.grad.ravel()).reshape(data.x.shape)  # todo: banded
        # scale attempted step if it is too large
        h = self.scale(h)
        # take step
        data.move(h, self.update)
        log.info(f'step ={norm(h):.5e}')

    def search(self, data, gtol=1e-5, maxiter=100, callback=None):
        """
        Return optimized coordinate from initial guess, data (an instance of Data)
        gtol    -- converged only if RMS gradient < gtol
        maxiter -- maximum number of overall iterations
        callback -- a user-supplied function called as callback(x,y) after each iteration
        """
        xt = [data.x.copy()]
        n_digit = int(np.log10(maxiter)) + 1

        for i in range(maxiter):
            log.info(f'iter {i:{n_digit}}: {data}')

            # check for convergence
            if norm(data.grad) < gtol:
                log.info(f'converged after {i} steps')
                break
            # update data by one iteration
            self.iterate(data)
            xt.append(data.x.copy())
            if callback:
                callback(data)

            log.debug(f'new x = {data.x}')
            log.debug(f'new G = {data.grad}')
            log.debug(f'new H = {data.hess}')

        else:
            log.warning('WARNING: did not converge')

        data.xt = np.array(xt)


class ModeFollowing(NewtonRaphson):
    """Following Wales, The Journal of Chemical Physics 101, 3750 (1994)"""
    def __init__(self, order=1, maxstep=None, project=None):
        super().__init__(maxstep, project)
        self.order = order

    def step(self, f, b):
        """Return step in eigenmodes"""
        sign = - np.ones_like(b)  # negative for minimization
        sign[:self.order] = 1  # positive for maximization
        return sign * 2 * f / (abs(b) * (1 + np.sqrt(1 + 4 * f ** 2 / b ** 2)))

    def iterate(self, data):
        """Take one iteration, including rescaling step"""
        # compute attempted step
        hess = data.hess.copy()

        # todo: project
        n_zero = 0  # todo
        b, eig_vecs = linalg.eigh(hess)  # todo: banded
        n = sum(b[np.argpartition(abs(b), n_zero)[n_zero:]] < 0)  # number of negative eigenvalues

        message = f'{n} -ve eigvals'
        if self.project:
            message += f' ({n_zero} zeros projected out)'
        log.info(message)

        f = np.dot(data.grad.ravel(), eig_vecs)  # f[i] is component of gradient along eigenvector[:,i]
        h = self.step(f, b)
        # scale attempted step if it is too large
        h = self.scale(h)
        h = np.dot(eig_vecs, h).reshape(data.x.shape)
        # take step
        data.move(h, self.update)
        log.info(f'step ={norm(h):.5e}')
        log.debug(f'eigvals: {b}')
        return data
