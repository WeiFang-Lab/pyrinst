import logging
import numpy as np
from numpy.linalg import norm
from numpy.typing import NDArray
from scipy import linalg
from pyrinst.core import Data

log = logging.getLogger(__name__)


class NewtonRaphson:
    """Base class of all quasi-newton optimizers.
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


class LBFGS(NewtonRaphson):
    """Limited-memory BFGS optimizer."""
    def __init__(self, maxstep: float = 0.3, **_):
        """Initializes the LBFGS optimizer.

        Parameters
        ----------
        maxstep : float, optional
            The maximum allowed step size for each iteration. Defaults to 0.3.
        """
        super().__init__(maxstep=maxstep)
        self.m: int = 3  # The number of previous steps and gradients to store.
        if self.m <= 0:
            raise ValueError('m must be a positive integer')
        self.dguess: float = 1  # Initial guess for the diagonal of the inverse Hessian approximation.
        self.wss: NDArray = np.zeros(self.m)
        self.wgd: NDArray = np.zeros(self.m)
        self.rho: NDArray = np.zeros(self.m)
        self.iter_num: int = 0

    def iterate(self, data: Data):
        """Performs a single L-BFGS iteration.

        This method computes the search direction using the L-BFGS two-loop
        recursion, scales the step, updates the position, and then updates
        the history of steps and gradient differences.

        Parameters
        ----------
        data : Data
            A `Data` object containing the current optimization state. Its `x`
            and `grad` attributes are used and updated.
        """
        g = data.grad.ravel()

        # 1. Compute the search direction q = -H_k * g_k
        q = -g
        alpha = np.zeros(self.m)

        # First loop (backward)
        for i in range(min(self.m, self.iter_num)):
            idx = (self.iter_num - 1 - i) % self.m
            alpha[idx] = self.rho[idx] * np.dot(self.wss[idx], q)
            q -= alpha[idx] * self.wgd[idx]

        # 2. Scale the direction with the initial Hessian approximation
        if self.iter_num > 0:
            prev_idx = (self.iter_num - 1) % self.m
            ys = np.dot(self.wgd[prev_idx], self.wss[prev_idx])
            yy = np.dot(self.wgd[prev_idx], self.wgd[prev_idx])
            if yy > 0:
                gamma = ys / yy
                q *= gamma
        else:
            q *= self.dguess

        # Second loop (forward)
        for i in range(min(self.m, self.iter_num) - 1, -1, -1):
            idx = (self.iter_num - 1 - i) % self.m
            beta = self.rho[idx] * np.dot(self.wgd[idx], q)
            q += self.wss[idx] * (alpha[idx] - beta)

        # 3. Determine step size and update position
        h = self.scale(q.reshape(data.x.shape))

        # Store current gradient for next iteration's difference calculation
        g_old = data.grad.copy()

        # Move to the new position
        data.move(h, update=True)
        log.info(f'step ={norm(h):.5e}')

        # 4. Store the new step (s) and gradient difference (y)
        if norm(data.grad - g_old) > 1e-8:  # Avoid division by zero
            s = h.ravel()
            y = (data.grad - g_old).ravel()

            self.wss[self.iter_num % self.m] = s
            self.wgd[self.iter_num % self.m] = y
            self.rho[self.iter_num % self.m] = 1.0 / np.dot(y, s)

        self.iter_num += 1

    def search(self, data: Data, gtol=1e-5, maxiter=100, callback=None):
        """Runs the L-BFGS optimization algorithm.

        Parameters
        ----------
        data : Data
            A `Data` object that provides the current state of the optimization,
            including position `x` and gradient `grad`. It is updated
            in-place.
        gtol : float, optional
            The tolerance for the gradient norm. The optimization is considered
            converged when `norm(data.grad) < gtol`. Defaults to 1e-5.
        maxiter : int, optional
            The maximum number of iterations to perform. Defaults to 100.
        callback : callable, optional
            A function to be called after each iteration. It receives the `data`
            object as its only argument. Defaults to None.
        """
        # Initialize storage arrays
        # wss and wgd are in circular order controlled by a pointer
        self.wss = np.zeros((self.m,) + data.x.shape)  # last m search steps
        self.wgd = np.zeros((self.m,) + data.x.shape)  # last m gradient differences
        self.rho = np.zeros(self.m)
        self.iter_num = 0
        super().search(data, gtol, maxiter, callback)


optimizers: dict[str, type] = {'EF': ModeFollowing, 'lBFGS': LBFGS}
