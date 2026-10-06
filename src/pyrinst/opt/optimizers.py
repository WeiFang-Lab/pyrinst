import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.linalg import norm
from numpy.typing import NDArray
from scipy import linalg

from pyrinst.potentials import Executor, Level

from .hessian import bfgs, bofill, half_ring_bands, powell, update_bead_hessians, update_bead_hessians_rigid
from .projections import centroid, proj_eig, rigid_body_basis
from .sbw_resolvent import ResolventError, banded_projected_sbw_step, banded_sbw_step

if TYPE_CHECKING:
    from pyrinst.geometries import StationaryPoint

log = logging.getLogger(__name__)

OPTIMIZER_REGISTRY: dict[str, type["NewtonRaphson"]] = {}


class NewtonRaphson:
    """Base class of all quasi-newton optimizers.
    The standard Newton-Raphson ignores argument order and just optimizes to any nearby stationary point.
    """

    type_alias: str = "NR"

    def __init__(self, executor: Executor, maxstep=None, project: bool = True, update: bool = True):
        """
        verbosity -- controls messages
        """
        self.executor = executor
        self.maxstep = maxstep
        self.project: bool = project
        self.update_method: Callable | None = bofill if update else None

    def __init_subclass__(cls):
        if cls.type_alias is not None:
            OPTIMIZER_REGISTRY[cls.type_alias] = cls

    def scale(self, h: NDArray) -> NDArray:
        if self.maxstep is not None:
            step = norm(h)
            if step > self.maxstep:
                h *= self.maxstep / step
        return h

    def move(self, data: "StationaryPoint", h: NDArray) -> None:
        if data.type_alias == "centroid":
            h -= h.mean(axis=0)
        x, G, H = data.x.copy(), data.G.copy(), data.H.copy()
        data.x += h
        if self.update_method:
            self.executor.compute(data, level=Level.GRAD)
            data.H = self.update_method(H, (data.x - x).ravel(), (data.G - G).ravel())
        else:
            self.executor.compute(data, level=Level.FREQ)

    def iterate(self, data: "StationaryPoint") -> None:
        """Take one iteration, including rescaling step"""
        # compute attempted step
        hess = data.H.copy()
        h = -linalg.solve(hess, data.G.ravel()).reshape(data.x.shape)  # todo: banded
        # scale attempted step if it is too large
        h = self.scale(h)
        # take step
        self.move(data, h)
        log.info(f"step ={norm(h):.5e}")

    def search(self, data: "StationaryPoint", gtol=1e-5, maxiter=100, callback=None):
        """
        Return optimized coordinate from initial guess, data (an instance of Data)
        gtol    -- converged only if RMS gradient < gtol
        maxiter -- maximum number of overall iterations
        callback -- a user-supplied function called as callback(x,y) after each iteration
        """
        self.executor.compute(data, level=Level.FREQ)  # todo: prevent redundant computation
        xt = [data.x.copy()]
        n_digit = int(np.log10(maxiter)) + 1
        for i in range(maxiter):
            log.info(f"iter {i:{n_digit}}: {data}")

            # check for convergence
            if norm(data.G) < gtol:
                log.info(f"converged after {i} steps")
                break
            # update data by one iteration
            self.iterate(data)
            xt.append(data.x.copy())
            if callback:
                callback(data)

            if log.isEnabledFor(logging.DEBUG):
                log.debug("new x = %s", data.x)
                log.debug("new G = %s", data.G)
                log.debug("new H = %s", data.H)

        else:
            log.warning("WARNING: did not converge")
        self.executor.compute(data, level=Level.FREQ)

        # data.xt = np.array(xt)


class ModeFollowing(NewtonRaphson):
    """Following Wales, The Journal of Chemical Physics 101, 3750 (1994)"""

    type_alias = "EF"

    def __init__(self, order: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.order: int = order

    def step(self, f: NDArray, b: NDArray) -> NDArray:
        """Return step in eigenmodes"""
        sign = -np.ones_like(b)  # negative for minimization
        sign[: self.order] = 1  # positive for maximization
        return sign * 2 * f / (abs(b) * (1 + np.sqrt(1 + 4 * f**2 / b**2)))

    def _eigenpairs(self, data: "StationaryPoint") -> tuple[NDArray, NDArray]:
        hess = data.H.copy()
        if data.type_alias == "centroid":
            b, eig_vecs = proj_eig(data.x, hess, 0, constr_vecs=centroid(data.x))
        elif self.project:
            b, eig_vecs = proj_eig(data.x, hess, data.n_zero, mass=data.m)
        else:
            b, eig_vecs = linalg.eigh(hess)  # todo: banded
            idx: NDArray = np.sort(np.argpartition(abs(b), data.n_zero)[data.n_zero :])
            b = b[idx]
            eig_vecs = eig_vecs[:, idx]
        return b, eig_vecs

    def iterate(self, data: "StationaryPoint") -> None:
        """Take one iteration, including rescaling step"""
        b, eig_vecs = self._eigenpairs(data)
        if not len(b):
            return
        n = sum(b < 0)  # number of negative eigenvalues

        message = f"{n} -ve eigvals"
        if self.project:
            message += f" ({data.n_zero} zeros projected out)"
        log.info(message)

        f = np.dot(data.G.ravel(), eig_vecs)  # f[i] is component of gradient along eigenvector[:,i]
        h = self.step(f, b)
        # scale attempted step if it is too large
        h = self.scale(h)
        h = np.dot(eig_vecs, h).reshape(data.x.shape)
        # take step
        self.move(data, h)
        log.info(f"step ={norm(h):.5e}")
        log.debug(f"eigvals: {b}")


class _BandedBeadHessian:
    """Keep bead potential Hessians local while reusing the parent's step rule."""

    def _eigenpairs(self, data: "StationaryPoint") -> tuple[NDArray, NDArray]:
        if (
            self.project
            or getattr(data, "type_alias", None) != "inst"
            or getattr(data, "full_ring", False)
            or data.hess is None
            or data.hess.ndim != 3
        ):
            return super()._eigenpairs(data)
        bands = half_ring_bands(data.hess, data.masses, data.springs.omega_n, data.x.shape[-1])
        b, eig_vecs = linalg.eig_banded(bands, lower=True)
        idx: NDArray = np.sort(np.argpartition(abs(b), data.n_zero)[data.n_zero :])
        return b[idx], eig_vecs[:, idx]

    def move(self, data: "StationaryPoint", h: NDArray) -> None:
        if not self.update_method or getattr(data, "type_alias", None) not in ("inst", "centroid"):
            super().move(data, h)
            return
        if data.hess is None or data.hess.ndim != 3:
            raise ValueError("bead-local Hessian update requires per-bead Hessians")
        if data.type_alias == "centroid":
            h -= h.mean(axis=0)
        x, grad, hess = data.x.copy(), data.grad.copy(), data.hess.copy()
        data.x += h
        self.executor.compute(data, level=Level.GRAD)
        method = "powell" if self.update_method is powell else "bfgs" if self.update_method is bfgs else "bofill"
        if getattr(self, "rigid_update", False):
            data.hess = update_bead_hessians_rigid(hess, data.x, data.grad, data.x - x, data.grad - grad)
        else:
            data.hess = update_bead_hessians(hess, data.x - x, data.grad - grad, method=method)


class BandedModeFollowing(_BandedBeadHessian, ModeFollowing):
    """Mode following with local Hessian updates and half-ring banded eigenpairs."""

    type_alias = "EFB"


class LBFGS(NewtonRaphson):
    """Limited-memory BFGS optimizer."""

    type_alias = "lBFGS"

    def __init__(self, executor: Executor, maxstep: float = 0.3, **_):
        """Initializes the LBFGS optimizer.

        Parameters
        ----------
        executor : Executor
            The executor used to evaluate the potential-energy surface.
        maxstep : float, optional
            The maximum allowed step size for each iteration. Defaults to 0.3.
        """
        super().__init__(executor, maxstep=maxstep)
        self.m: int = 3  # The number of previous steps and gradients to store.
        if self.m <= 0:
            raise ValueError("m must be a positive integer")
        self.dguess: float = 1  # Initial guess for the diagonal of the inverse Hessian approximation.
        self.wss: NDArray = np.zeros(self.m)
        self.wgd: NDArray = np.zeros(self.m)
        self.rho: NDArray = np.zeros(self.m)
        self.iter_num: int = 0

    def iterate(self, data: "StationaryPoint") -> None:
        """Performs a single L-BFGS iteration.

        This method computes the search direction using the L-BFGS two-loop
        recursion, scales the step, updates the position, and then updates
        the history of steps and gradient differences.

        Parameters
        ----------
        data : StationaryPoint
            A `StationaryPoint` object containing the current optimization state. Its `x`
            and `G` attributes are used and updated.
        """
        g = data.G.ravel()

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
        g_old = data.G.copy()

        # Move to the new position
        self.move(data, h)
        log.info(f"step ={norm(h):.5e}")

        # 4. Store the new step (s) and gradient difference (y)
        if norm(data.G - g_old) > 1e-8:  # Avoid division by zero
            s = h.ravel()
            y = (data.G - g_old).ravel()

            self.wss[self.iter_num % self.m] = s
            self.wgd[self.iter_num % self.m] = y
            self.rho[self.iter_num % self.m] = 1.0 / np.dot(y, s)

        self.iter_num += 1

    def search(self, data: "StationaryPoint", gtol=1e-5, maxiter=100, callback=None):
        """Runs the L-BFGS optimization algorithm.

        Parameters
        ----------
        data : StationaryPoint
            A `StationaryPoint` object that provides the current state of the optimization,
            including position `x` and gradient `G`. It is updated
            in-place.
        gtol : float, optional
            The tolerance for the gradient norm. The optimization is considered
            converged when `norm(data.G) < gtol`. Defaults to 1e-5.
        maxiter : int, optional
            The maximum number of iterations to perform. Defaults to 100.
        callback : callable, optional
            A function to be called after each iteration. It receives the `data`
            object as its only argument. Defaults to None.
        """
        # Initialize storage arrays
        # wss and wgd are in circular order controlled by a pointer
        self.wss = np.zeros((self.m, data.x.size))  # last m search steps
        self.wgd = np.zeros((self.m, data.x.size))  # last m gradient differences
        self.rho = np.zeros(self.m)
        self.iter_num = 0
        super().search(data, gtol, maxiter, callback)


class StreamBedWalk(ModeFollowing):
    """
    J. Chem. Phys. 1990, 92 (1), 340-346.
    Walks from x0 to a minimum (order=0), transition state (order=1) or other saddle point (order>1)
    of the potential energy surface.
    """

    type_alias = "SBW"

    def __init__(self, update: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_method = (bfgs if self.order == 0 else powell) if update else None

    def step(self, f, b):
        if self.order == 0:
            xv = -f / b
            alpha = 1
            lam = b[0] - abs(f[0] / self.maxstep) if b[0] < 0 or norm(xv) > self.maxstep else 0

        else:
            # invert sign in cases of order>1 only
            b[1 : self.order] *= -1
            f[1 : self.order] *= -1
            alpha, lam = self.alpha_lambda(b[0], b[1])
        return alpha * f / (lam - b)  # step in ev space

    @staticmethod
    def alpha_lambda(b0: float, b1: float) -> tuple[float, float]:
        """Use the original SBW scalar rule for saddle searches."""
        if b0 > 0:
            if 0.5 * b1 > b0:
                return 1.0, 0.5 * (b0 + 0.5 * b1)
            return (b1 - b0) / b1, 0.25 * (3 * b0 + b1)
        if b1 < 0:
            if b1 >= 0.5 * b0:
                return 1.0, 0.5 * (0.5 * b0 + b1)
            return (b0 - b1) / b1, 0.25 * (b0 + 3 * b1)
        return 1.0, 0.25 * (b0 + b1)


class BandedStreamBedWalk(_BandedBeadHessian, StreamBedWalk):
    """SBW with local Hessian updates and half-ring banded eigenpairs."""

    type_alias = "SBWB"


class BandedResolventStreamBedWalk(_BandedBeadHessian, StreamBedWalk):
    """SBW with bead-local Hessian updates and a banded resolvent step.

    Supports projected or unprojected first-order half-ring instantons.
    Failed numerical checks fall back to the matching dense SBW step.
    """

    type_alias = "SBWR"

    def __init__(
        self,
        *args,
        projection_metric: str = "cartesian",
        low_mode_shift: str = "eigenvalue",
        low_mode_preconditioner: str = "projected",
        band_lu: bool = True,
        warm_start: bool = True,
        warm_shift_mode: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if projection_metric not in ("cartesian", "legacy_mass"):
            raise ValueError("projection metric must be cartesian or legacy_mass")
        if low_mode_shift not in ("eigenvalue", "bound"):
            raise ValueError("low-mode shift must be eigenvalue or bound")
        if low_mode_preconditioner not in ("projected", "constrained"):
            raise ValueError("low-mode preconditioner must be projected or constrained")
        self.projection_metric = projection_metric
        self.low_mode_shift = low_mode_shift
        self.low_mode_preconditioner = low_mode_preconditioner
        self.band_lu = band_lu
        self.warm_start = warm_start
        self.warm_shift_mode = warm_shift_mode
        self.low_mode_vectors: NDArray | None = None
        self.shift_mode_vector: NDArray | None = None
        self.low_mode_history_lengths: list[int] = []
        self.warm_start_retries = 0
        self.resolvent_seconds = 0.0
        self.resolvent_calls = 0
        self.resolvent_fallbacks = 0
        self.resolvent_fallback_iterations: list[int] = []
        self.resolvent_failure_reasons: list[str] = []
        self.resolvent_component_seconds: dict[str, float] = {}

    def _checked_displacement(self, step: NDArray, constraints: NDArray | None) -> NDArray:
        """Reject invalid coordinates before moving or updating bead Hessians."""
        if not np.all(np.isfinite(step)):
            raise ResolventError("SBW displacement is not finite")
        if not np.isfinite(norm(step)):
            raise ResolventError("SBW displacement norm is not finite")
        step = self.scale(step)
        step_norm = norm(step)
        if not np.isfinite(step_norm):
            raise ResolventError("SBW displacement norm is not finite")
        if self.maxstep is not None and step_norm > self.maxstep * (1 + 1e-12):
            raise ResolventError("SBW displacement exceeds maxstep")
        if constraints is not None and norm(constraints.T @ step) > 1e-8 * max(1.0, step_norm):
            raise ResolventError("SBW displacement violates rigid-body constraints")
        return step

    def iterate(self, data: "StationaryPoint") -> None:
        if (
            self.order != 1
            or getattr(data, "type_alias", None) != "inst"
            or getattr(data, "full_ring", False)
            or data.hess is None
            or data.hess.ndim != 3
        ):
            raise NotImplementedError("SBWR requires a first-order half-ring instanton")
        bands = half_ring_bands(data.hess, data.masses, data.springs.omega_n, data.x.shape[-1])
        constraints = (
            rigid_body_basis(data.x, data.n_zero, data.masses, self.projection_metric) if self.project else None
        )
        began = time.perf_counter()
        self.resolvent_calls += 1
        try:
            if self.project:
                # The spring Hessian is positive semidefinite, so twice the
                # lowest bead potential curvature bounds the half-ring below.
                lower_bound = 2 * float(np.min(np.linalg.eigvalsh(data.hess)))
                initial = self.low_mode_vectors if self.warm_start else None
                if initial is not None and initial.shape[0] != bands.shape[1]:
                    initial = None
                shift_initial = self.shift_mode_vector if self.warm_start and self.warm_shift_mode else None
                if shift_initial is not None and shift_initial.shape != (bands.shape[1],):
                    shift_initial = None
                diagnostics: dict = {}
                try:
                    h, timings = banded_projected_sbw_step(
                        bands, data.G.ravel(), constraints, self.alpha_lambda,
                        potential_lower_bound=lower_bound, initial_vectors=initial,
                        initial_shift_vector=shift_initial,
                        diagnostics=diagnostics, shift_selection=self.low_mode_shift,
                        preconditioner=self.low_mode_preconditioner, band_lu=self.band_lu,
                    )
                except (ResolventError, linalg.LinAlgError, RuntimeError, ValueError):
                    if initial is None:
                        raise
                    self.warm_start_retries += 1
                    diagnostics = {}
                    h, timings = banded_projected_sbw_step(
                        bands, data.G.ravel(), constraints, self.alpha_lambda,
                        potential_lower_bound=lower_bound, diagnostics=diagnostics,
                        shift_selection=self.low_mode_shift,
                        preconditioner=self.low_mode_preconditioner, band_lu=self.band_lu,
                    )
            else:
                h, timings = banded_sbw_step(bands, data.G.ravel(), data.n_zero, self.alpha_lambda)
            h = self._checked_displacement(h, constraints)
            if self.project:
                if self.warm_start:
                    self.low_mode_vectors = diagnostics["low_vectors"]
                    self.shift_mode_vector = diagnostics["shift_vector"] if self.warm_shift_mode else None
                self.low_mode_history_lengths.append(diagnostics["lobpcg_history_length"])
            for name, seconds in timings.items():
                self.resolvent_component_seconds[name] = self.resolvent_component_seconds.get(name, 0.0) + seconds
        except (ResolventError, linalg.LinAlgError, RuntimeError, ValueError) as exc:
            self.low_mode_vectors = None
            self.shift_mode_vector = None
            self.resolvent_fallbacks += 1
            self.resolvent_fallback_iterations.append(self.resolvent_calls)
            self.resolvent_failure_reasons.append(str(exc))
            log.warning("SBWR falling back to dense eigensolver: %s", exc)
            hess = data.H.copy()
            if self.project:
                projector = np.eye(len(hess)) - constraints @ constraints.T
                hess = projector @ hess @ projector
            b, eig_vecs = linalg.eigh(hess)
            keep = np.sort(np.argpartition(abs(b), data.n_zero)[data.n_zero :])
            f = data.G.ravel() @ eig_vecs[:, keep]
            h = self._checked_displacement(eig_vecs[:, keep] @ self.step(f, b[keep]), constraints)
        self.resolvent_seconds += time.perf_counter() - began
        h = h.reshape(data.x.shape)
        self.move(data, h)
        log.info(f"step ={norm(h):.5e}")


class RigidBandedResolventStreamBedWalk(BandedResolventStreamBedWalk):
    """First-order half-ring SBW with rigid-covariant bead Hessian updates.

    The PES must be translation and rotation invariant in Cartesian space.
    Its gradients are checked for compatibility with a symmetric Hessian at
    every updated bead. The spring block and band topology are unchanged.
    """

    type_alias = "SBWRP"
    rigid_update = True
