"""Banded resolvent steps for first-order SBW saddle searches."""

import time
import warnings
from collections.abc import Callable

import numpy as np
from numpy.linalg import norm
from numpy.typing import NDArray
from scipy import linalg, sparse
from scipy.sparse.linalg import LinearOperator, eigsh, lobpcg, splu


class ResolventError(RuntimeError):
    """The structured solve did not pass its numerical checks."""


def _sparse_from_lower_bands(bands: NDArray) -> sparse.csc_matrix:
    width = len(bands) - 1
    size = bands.shape[1]
    diagonals = [bands[0]]
    offsets = [0]
    for offset in range(1, width + 1):
        values = bands[offset, : size - offset]
        diagonals.extend((values, values))
        offsets.extend((-offset, offset))
    return sparse.diags(diagonals, offsets, shape=(size, size), format="csc")


def _shifted_general_bands(bands: NDArray, shift: float) -> NDArray:
    width = len(bands) - 1
    size = bands.shape[1]
    result = np.zeros((2 * width + 1, size), dtype=bands.dtype)
    result[width] = shift - bands[0]
    for offset in range(1, width + 1):
        values = -bands[offset, : size - offset]
        result[width + offset, : size - offset] = values
        result[width - offset, offset:] = values
    return result


def _band_spectral_bounds(bands: NDArray) -> tuple[float, float]:
    """Return a Gershgorin lower bound and an absolute spectral bound."""
    size = bands.shape[1]
    off_diagonal = np.zeros(size, dtype=bands.dtype)
    for offset in range(1, len(bands)):
        entries = abs(bands[offset, : size - offset])
        off_diagonal[: size - offset] += entries
        off_diagonal[offset:] += entries
    lower_bound = float(np.min(bands[0] - off_diagonal))
    scale = max(1.0, float(np.max(abs(bands[0]) + off_diagonal)))
    return lower_bound, scale


def _lowest_band_eigenvalue(bands: NDArray, matrix: sparse.csc_matrix, lower_bound: float, scale: float) -> float:
    """Find the lowest eigenvalue with a shift below the whole spectrum."""
    shift = lower_bound - max(0.01, 0.1 * abs(lower_bound))
    shifted = bands.copy()
    shifted[0] -= shift
    factor = linalg.cholesky_banded(shifted, lower=True)
    inverse = LinearOperator(matrix.shape, matvec=lambda vector: linalg.cho_solve_banded((factor, True), vector))
    values, vectors = eigsh(
        matrix,
        k=1,
        sigma=shift,
        which="LM",
        OPinv=inverse,
        v0=np.random.default_rng(17).normal(size=matrix.shape[0]),
        tol=1e-10,
    )
    residual = norm(matrix @ vectors[:, 0] - values[0] * vectors[:, 0])
    if not np.isfinite(values[0]) or values[0] < lower_bound - 1e-8 * scale or residual > 1e-8 * scale:
        raise ResolventError(f"lowest band eigenpair failed checks: residual={residual:.3e}")
    return float(values[0])


def _shifted_band_lu(bands: NDArray, shift: float) -> Callable[[NDArray], NDArray]:
    """Factor H - shift*I in LAPACK band storage, including pivot fill."""
    width = len(bands) - 1
    size = bands.shape[1]
    packed = np.zeros((3 * width + 1, size), dtype=bands.dtype, order="F")
    packed[2 * width] = bands[0] - shift
    for offset in range(1, width + 1):
        entries = bands[offset, : size - offset]
        packed[2 * width + offset, : size - offset] = entries
        packed[2 * width - offset, offset:] = entries
    gbtrf, gbtrs = linalg.lapack.get_lapack_funcs(("gbtrf", "gbtrs"), (packed,))
    factor, pivots, info = gbtrf(packed, width, width, overwrite_ab=True)
    if info:
        raise linalg.LinAlgError(f"shifted band LU failed: info={info}")

    def solve(rhs: NDArray) -> NDArray:
        vector = np.asarray(rhs)
        one_dimensional = vector.ndim == 1
        result, solve_info = gbtrs(factor, width, width, vector[:, None] if one_dimensional else vector, pivots)
        if solve_info:
            raise linalg.LinAlgError(f"shifted band LU solve failed: info={solve_info}")
        return result[:, 0] if one_dimensional else result

    return solve


def _nearest_band_eigenvalue(
    matrix: sparse.csc_matrix, shift: float, scale: float, bands: NDArray | None = None,
    *, initial_vector: NDArray | None = None, diagnostics: dict | None = None,
) -> tuple[float, Callable[[NDArray], NDArray]]:
    """Check that the SBW shift is separated from the full band spectrum."""
    if bands is None:
        factor = splu(matrix - shift * sparse.eye(matrix.shape[0], format="csc"), permc_spec="NATURAL")
        solve = factor.solve
    else:
        solve = _shifted_band_lu(bands, shift)
    if initial_vector is not None:
        initial_vector = np.asarray(initial_vector)
        if (
            initial_vector.shape != (matrix.shape[0],)
            or not np.all(np.isfinite(initial_vector))
            or norm(initial_vector) == 0
        ):
            raise ValueError("shift-mode initial vector has invalid shape or values")
    inverse = LinearOperator(matrix.shape, matvec=solve, dtype=float)
    values, vectors = eigsh(matrix, k=1, sigma=shift, which="LM", OPinv=inverse, v0=initial_vector, tol=1e-10)
    residual = norm(matrix @ vectors[:, 0] - values[0] * vectors[:, 0])
    if (
        not np.isfinite(values[0])
        or not np.isfinite(residual)
        or abs(values[0] - shift) < 1e-10 * scale
        or residual > 1e-8 * scale
    ):
        raise ResolventError(f"SBW shift is singular or too close to the spectrum: residual={residual:.3e}")
    if initial_vector is not None:
        # An old eigenvector can remain an exact eigenvector after another
        # mode crosses the shift. An independent inverse action detects any
        # missed eigenvalue whose response exceeds the returned Ritz value.
        probe = np.random.default_rng(7919).normal(size=matrix.shape[0])
        probe /= norm(probe)
        probe_norm = norm(solve(probe))
        if not np.isfinite(probe_norm) or probe_norm > 1.01 / abs(values[0] - shift):
            raise ResolventError("warm shift-mode start missed a closer eigenvalue")
    if diagnostics is not None:
        diagnostics["shift_vector"] = vectors[:, 0].copy()
        diagnostics["shift_eigenvalue"] = float(values[0])
    return float(values[0]), solve


def _zero_mode_shift(values: NDArray, omitted: NDArray) -> float:
    """Choose a shift whose closest eigenvalues are precisely the omitted block."""
    if not np.array_equal(omitted, np.arange(omitted[0], omitted[-1] + 1)):
        raise ResolventError("excluded modes do not occupy a contiguous spectral block")
    first, last = int(omitted[0]), int(omitted[-1])
    if first == 0 or last == len(values) - 1:
        raise ResolventError("excluded modes lack both neighboring eigenvalues")
    low = 0.5 * (values[first - 1] + values[last])
    high = 0.5 * (values[last + 1] + values[first])
    if not low < high:
        raise ResolventError("no shift selects exactly the excluded modes")
    # The center of the target block maximizes separation from its farthest
    # member. The midpoint of the admissible interval can be far outside a
    # near-zero cluster when the negative saddle mode is much farther away.
    shift = 0.5 * (values[first] + values[last])
    scale = max(1.0, float(np.max(abs(values))))
    if np.min(abs(values - shift)) < 1e-10 * scale:
        # A single omitted mode has no internal gap. Move the shift toward
        # either edge of its admissible interval instead of always falling
        # back to a dense eigensolve. The same candidates help a tight cluster.
        candidates = [
            0.5 * (low + values[first]),
            0.5 * (values[last] + high),
            *(0.5 * (values[index] + values[index + 1]) for index in range(first, last)),
        ]
        candidates = [candidate for candidate in candidates if low < candidate < high]
        if not candidates:
            raise ResolventError("zero-mode shift is too close to the spectrum")
        shift = max(candidates, key=lambda candidate: np.min(abs(values - candidate)))
        nearest = np.sort(np.argsort(abs(values - shift))[: len(omitted)])
        if np.min(abs(values - shift)) < 1e-10 * scale or not np.array_equal(nearest, omitted):
            raise ResolventError("zero-mode shift is too close to the spectrum")
    return float(shift)


def banded_sbw_step(bands: NDArray, gradient: NDArray, n_zero: int, alpha_lambda) -> tuple[NDArray, dict[str, float]]:
    """Return the exact SBW mode step without forming all eigenvectors.

    ``alpha_lambda`` is the same scalar rule used by the ordinary SBW step.
    The caller handles maximum-step scaling and the bead-local Hessian update.
    """
    size = bands.shape[1]
    if not 0 <= n_zero < size - 1 or norm(gradient) == 0:
        raise ResolventError("invalid mode count or zero gradient")
    if not np.all(np.isfinite(bands)) or not np.all(np.isfinite(gradient)):
        raise ResolventError("nonfinite Hessian or gradient")
    timings: dict[str, float] = {}
    began = time.perf_counter()
    values = linalg.eig_banded(bands, lower=True, eigvals_only=True)
    timings["values_seconds"] = time.perf_counter() - began
    if n_zero:
        partition = np.argpartition(abs(values), n_zero)
        omitted = np.sort(partition[:n_zero])
        retained = np.sort(partition[n_zero:])
    else:
        omitted = np.empty(0, dtype=int)
        retained = np.arange(size)
    alpha, shift = alpha_lambda(values[retained][0], values[retained][1])
    scale = max(1.0, float(np.max(abs(values))))
    if not np.isfinite(alpha) or not np.isfinite(shift) or np.min(abs(values - shift)) < 1e-10 * scale:
        raise ResolventError("SBW step shift is singular or too close to the spectrum")

    began = time.perf_counter()
    if n_zero:
        zero_shift = _zero_mode_shift(values, omitted)
        matrix = _sparse_from_lower_bands(bands)
        factor = splu(matrix - zero_shift * sparse.eye(size, format="csc"), permc_spec="NATURAL")
        inverse = LinearOperator((size, size), matvec=factor.solve, dtype=float)
        near_values, zero_vectors = eigsh(
            matrix,
            k=n_zero,
            sigma=zero_shift,
            which="LM",
            OPinv=inverse,
            tol=1e-10,
            ncv=min(size, max(2 * n_zero + 12, 20)),
            maxiter=max(1000, size),
        )
        spectral_error = float(np.max(abs(np.sort(near_values) - values[omitted])))
        vector_residual = float(np.max(norm(matrix @ zero_vectors - zero_vectors * near_values, axis=0)))
        orthogonality_error = float(norm(zero_vectors.T @ zero_vectors - np.eye(n_zero)))
        if spectral_error > 1e-8 * scale or vector_residual > 1e-8 * scale or orthogonality_error > 1e-8:
            raise ResolventError(
                "excluded eigenvectors failed checks: "
                f"spectral={spectral_error:.3e}, residual={vector_residual:.3e}, "
                f"orthogonality={orthogonality_error:.3e}, scale={scale:.3e}"
            )
        projected = gradient - zero_vectors @ (zero_vectors.T @ gradient)
    else:
        zero_vectors = np.empty((size, 0))
        matrix = _sparse_from_lower_bands(bands)
        projected = gradient
    timings["zero_modes_seconds"] = time.perf_counter() - began

    began = time.perf_counter()
    response = linalg.solve_banded((len(bands) - 1, len(bands) - 1), _shifted_general_bands(bands, shift), projected)
    residual = norm(shift * response - matrix @ response - projected)
    denominator = (abs(shift) + scale) * norm(response) + norm(projected)
    if not np.all(np.isfinite(response)) or residual > 1e-9 * denominator:
        raise ResolventError("shifted banded solve failed its residual check")
    response -= zero_vectors @ (zero_vectors.T @ response)
    timings["solve_seconds"] = time.perf_counter() - began
    timings["total_seconds"] = sum(timings.values())
    return alpha * response, timings


def banded_projected_sbw_step(
    bands: NDArray,
    gradient: NDArray,
    constraints: NDArray,
    alpha_lambda,
    potential_lower_bound: float | None = None,
    *,
    initial_vectors: NDArray | None = None,
    initial_shift_vector: NDArray | None = None,
    diagnostics: dict | None = None,
    shift_selection: str = "eigenvalue",
    preconditioner: str = "projected",
    band_lu: bool = True,
) -> tuple[NDArray, dict[str, float]]:
    """Solve the projected SBW step using only banded and low-rank operations."""
    size = bands.shape[1]
    if constraints.shape[0] != size or gradient.shape != (size,) or constraints.shape[1] >= size - 1:
        raise ResolventError("invalid projected SBW dimensions")
    if not np.all(np.isfinite(bands)) or not np.all(np.isfinite(gradient)):
        raise ResolventError("nonfinite Hessian or gradient")
    if not len(constraints.T):
        return banded_sbw_step(bands, gradient, 0, alpha_lambda)
    if norm(constraints.T @ constraints - np.eye(constraints.shape[1])) > 1e-8:
        raise ResolventError("projection constraints are not orthonormal")
    if shift_selection not in ("eigenvalue", "bound") or preconditioner not in ("projected", "constrained"):
        raise ValueError("unknown projected solver strategy")

    def project(vectors: NDArray) -> NDArray:
        return vectors - constraints @ (constraints.T @ vectors)

    timings: dict[str, float] = {}
    began = time.perf_counter()
    gershgorin_lower, scale = _band_spectral_bounds(bands)
    if potential_lower_bound is None:
        potential_lower_bound = gershgorin_lower
    matrix = _sparse_from_lower_bands(bands)
    lowest = (
        _lowest_band_eigenvalue(bands, matrix, potential_lower_bound, scale)
        if shift_selection == "eigenvalue"
        else min(gershgorin_lower, potential_lower_bound)
    )
    timings["minimum_eigen_seconds"] = time.perf_counter() - began
    began = time.perf_counter()
    tau = float(lowest - max(0.01, 0.1 * abs(lowest)))
    positive_bands = bands.copy()
    positive_bands[0] -= tau
    factor = linalg.cholesky_banded(positive_bands, lower=True)
    if preconditioner == "constrained":
        response_to_constraints = linalg.cho_solve_banded((factor, True), constraints)
        small = constraints.T @ response_to_constraints
        small = (small + small.T) / 2
        small_factor = linalg.cho_factor(small, lower=True)
    timings["preparation_seconds"] = time.perf_counter() - began

    def projected_hessian(vectors: NDArray) -> NDArray:
        return project(matrix @ project(vectors))

    def precondition(vectors: NDArray) -> NDArray:
        result = linalg.cho_solve_banded((factor, True), project(vectors))
        if preconditioner == "constrained":
            result -= response_to_constraints @ linalg.cho_solve(small_factor, constraints.T @ result)
        return project(result)

    began = time.perf_counter()
    initial = np.random.default_rng(17).normal(size=(size, 2))
    if initial_vectors is not None:
        trial = np.asarray(initial_vectors)
        if trial.shape != initial.shape or not np.all(np.isfinite(trial)):
            raise ValueError("initial vectors have invalid shape or values")
        trial = project(trial)
        basis, singular, _ = np.linalg.svd(trial, full_matrices=False)
        if singular[-1] > 1e-10 * singular[0]:
            initial = basis
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always", UserWarning)
        low_values, low_vectors, history = lobpcg(
            projected_hessian,
            initial,
            Y=constraints,
            M=precondition,
            largest=False,
            tol=1e-10,
            maxiter=300,
            retResidualNormsHistory=True,
        )
    timings["projected_eigen_seconds"] = time.perf_counter() - began
    order = np.argsort(low_values)
    low_values, low_vectors = low_values[order], low_vectors[:, order]
    eigen_residual = norm(projected_hessian(low_vectors) - low_vectors * low_values, axis=0)
    if (
        not np.all(np.isfinite(low_values))
        or np.max(eigen_residual) > 1e-9 * max(1.0, float(np.max(abs(low_values))))
        or norm(constraints.T @ low_vectors) > 1e-8
    ):
        warning = str(messages[-1].message).splitlines()[0] if messages else ""
        raise ResolventError(f"projected eigenpairs failed checks: residual={np.max(eigen_residual):.3e}; {warning}")
    if diagnostics is not None:
        diagnostics["low_vectors"] = low_vectors.copy()
        diagnostics["low_values"] = low_values.copy()
        diagnostics["lobpcg_history_length"] = len(history)
        diagnostics["lobpcg_residual"] = float(np.max(eigen_residual))
    alpha, shift = alpha_lambda(low_values[0], low_values[1])
    if not np.isfinite(alpha) or not np.isfinite(shift):
        raise ResolventError("projected SBW shift is not finite")

    began = time.perf_counter()
    _, shifted_solve = _nearest_band_eigenvalue(
        matrix, shift, scale, bands if band_lu else None,
        initial_vector=initial_shift_vector, diagnostics=diagnostics,
    )
    timings["shift_check_seconds"] = time.perf_counter() - began

    began = time.perf_counter()
    rhs = np.column_stack((gradient, constraints))
    solutions = (
        -shifted_solve(rhs)
        if band_lu
        else linalg.solve_banded((len(bands) - 1, len(bands) - 1), _shifted_general_bands(bands, shift), rhs)
    )
    y, response_to_constraints = solutions[:, 0], solutions[:, 1:]
    small_matrix = constraints.T @ response_to_constraints
    if np.linalg.cond(small_matrix) > 1e10:
        raise ResolventError("projected SBW Schur complement is ill-conditioned")
    multipliers = linalg.solve(small_matrix, constraints.T @ y, assume_a="sym")
    response = y - response_to_constraints @ multipliers
    equation_residual = norm(shift * response - matrix @ response + constraints @ multipliers - gradient)
    residual_scale = norm(gradient) + (abs(shift) + scale) * norm(response) + norm(multipliers)
    if (
        not np.all(np.isfinite(response))
        or equation_residual > 1e-9 * residual_scale
        or norm(constraints.T @ response) > 1e-9 * max(1.0, norm(response))
    ):
        raise ResolventError("projected SBW constrained solve failed its residual checks")
    timings["solve_seconds"] = time.perf_counter() - began
    timings["total_seconds"] = sum(timings.values())
    return alpha * response, timings
