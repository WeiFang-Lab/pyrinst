import numpy as np
import pytest
from scipy import linalg

from pyrinst.geometries import Instanton, InstRef
from pyrinst.opt.hessian import bfgs_hessian, half_ring_bands, powell, update_bead_hessians
from pyrinst.opt.optimizers import BandedModeFollowing, BandedResolventStreamBedWalk, BandedStreamBedWalk, StreamBedWalk
from pyrinst.opt.projections import proj_eig, rigid_body_basis, rot
from pyrinst.opt.sbw_resolvent import (
    ResolventError,
    _nearest_band_eigenvalue,
    _shifted_band_lu,
    _sparse_from_lower_bands,
    _zero_mode_shift,
    banded_projected_sbw_step,
    banded_sbw_step,
)
from pyrinst.potentials import Level


class QuarticExecutor:
    def compute(self, data, level=Level.GRAD):
        x = data.x[..., 0]
        grad = np.zeros_like(data.x)
        grad[..., 0] = x**3 - x
        grad[..., 1] = 1.3 * data.x[..., 1]
        grad[..., 2] = 1.7 * data.x[..., 2]
        hess = None
        if level == Level.FREQ:
            hess = np.zeros((len(x), 3, 3))
            hess[:, 0, 0] = 3 * x[:, 0] ** 2 - 1
            hess[:, 1, 1] = 1.3
            hess[:, 2, 2] = 1.7
        data.V = np.sum(
            0.25 * x**4 - 0.5 * x**2 + 0.65 * data.x[..., 1] ** 2 + 0.85 * data.x[..., 2] ** 2,
            axis=1,
        )
        data.G = grad
        data.H = hess


class ProjectionExecutor:
    def compute(self, data, level=Level.GRAD):
        diagonal = np.array([-1.0, 1.3, 1.7, -0.8, 1.2, 1.5])
        flat = data.x.reshape(len(data.x), -1)
        data.V = 0.5 * np.sum(diagonal * flat**2, axis=1)
        data.G = (diagonal * flat).reshape(data.x.shape)
        data.H = np.tile(np.diag(diagonal), (len(data.x), 1, 1)) if level == Level.FREQ else None


def make_instanton(cls=Instanton):
    x = np.zeros((8, 1, 3))
    x[:, 0, 0] = np.linspace(-0.8, 0.8, len(x))
    data = cls(x, ["H"], masses=np.array([1.0]), beta=25.0)
    QuarticExecutor().compute(data, Level.FREQ)
    return data


def make_projection_instanton():
    x = np.zeros((8, 2, 3))
    t = np.linspace(-0.8, 0.8, len(x))
    x[:, 0, 0] = t
    x[:, 1, 0] = 0.3 - 0.2 * t
    x[:, 1, 1] = 1 + 0.1 * t
    x[:, 1, 2] = 0.3 + 0.2 * t**2
    data = Instanton(x, ["H", "O"], masses=np.array([1.0, 16.0]), beta=25.0, n_zero=6)
    ProjectionExecutor().compute(data, Level.FREQ)
    return data


def unpack_lower_bands(bands):
    width = len(bands) - 1
    n = bands.shape[1]
    dense = np.zeros((n, n))
    for col in range(n):
        for offset in range(min(width, n - col - 1) + 1):
            dense[col + offset, col] = dense[col, col + offset] = bands[offset, col]
    return dense


def test_half_ring_bands_match_existing_hessian_and_eigenpairs():
    data = make_instanton()
    bands = half_ring_bands(data.hess, data.masses, data.springs.omega_n, 3)
    np.testing.assert_allclose(unpack_lower_bands(bands), data.H, atol=1e-13)
    opt = BandedModeFollowing(order=1, executor=QuarticExecutor(), project=False)
    eigenvalues, eigenvectors = opt._eigenpairs(data)
    np.testing.assert_allclose(eigenvalues, linalg.eigh(data.H, eigvals_only=True), atol=1e-12)
    np.testing.assert_allclose(data.H @ eigenvectors, eigenvectors * eigenvalues, atol=1e-12)


def test_half_ring_bands_multiple_atoms_and_unequal_masses():
    rng = np.random.default_rng(5)
    data = Instanton(np.zeros((4, 2, 3)), ["H", "O"], masses=np.array([1.0, 16.0]), beta=30.0)
    h = rng.normal(size=(4, 6, 6))
    data.hess = (h + h.transpose(0, 2, 1)) / 2
    bands = half_ring_bands(data.hess, data.masses, data.springs.omega_n, 3)
    np.testing.assert_allclose(unpack_lower_bands(bands), data.H, atol=1e-13)


def test_local_update_keeps_half_ring_banded_and_secant():
    data = make_instanton()
    before = data.G.copy()
    step = np.zeros_like(data.x)
    step[:, 0, 0] = np.linspace(-0.03, 0.04, len(step))
    opt = BandedModeFollowing(order=1, executor=QuarticExecutor(), project=False)
    opt.move(data, step)
    assert data.hess.shape == (8, 3, 3)
    bands = half_ring_bands(data.hess, data.masses, data.springs.omega_n, 3)
    np.testing.assert_allclose(unpack_lower_bands(bands), data.H, atol=1e-13)
    np.testing.assert_allclose(data.H @ step.ravel(), (data.G - before).ravel(), atol=1e-11)


def test_fixed_centroid_update_retains_bead_blocks_and_projected_secant():
    data = make_instanton(InstRef)
    before = data.G.copy()
    step = np.zeros_like(data.x)
    step[:, 0, 0] = np.linspace(-0.03, 0.03, len(step))
    opt = BandedModeFollowing(order=0, executor=QuarticExecutor(), project=False)
    opt.move(data, step)
    assert data.hess.shape == (8, 3, 3)
    np.testing.assert_allclose(data.H @ step.ravel(), (data.G - before).ravel(), atol=1e-11)
    eigenvalues, eigenvectors = opt._eigenpairs(data)
    np.testing.assert_allclose(data.H @ eigenvectors, eigenvectors * eigenvalues, atol=1e-11)


def test_sbw_saddle_uses_local_powell_and_banded_eigenpairs():
    data = make_instanton()
    before_x, before_g, before_h = data.x.copy(), data.grad.copy(), data.hess.copy()
    step = np.zeros_like(data.x)
    step[:, 0, 0] = np.linspace(-0.03, 0.04, len(step))
    opt = BandedStreamBedWalk(order=1, executor=QuarticExecutor(), project=False)
    opt.move(data, step)
    for i in range(len(step)):
        displacement = (data.x[i] - before_x[i]).ravel()
        expected = (
            powell(before_h[i].copy(), displacement, (data.grad[i] - before_g[i]).ravel())
            if np.linalg.norm(displacement) > 0
            else before_h[i]
        )
        np.testing.assert_allclose(data.hess[i], expected, atol=1e-12)
    bands = half_ring_bands(data.hess, data.masses, data.springs.omega_n, 3)
    np.testing.assert_allclose(unpack_lower_bands(bands), data.H, atol=1e-12)
    eigenvalues, eigenvectors = opt._eigenpairs(data)
    np.testing.assert_allclose(data.H @ eigenvectors, eigenvectors * eigenvalues, atol=1e-11)


def test_hessian_bfgs_and_sbw_minimum_local_update_satisfy_secant():
    h = np.diag([2.0, 3.0])
    step = np.array([0.1, 0.2])
    dg = np.array([0.3, 0.7])
    new_h = bfgs_hessian(h, step, dg)
    np.testing.assert_allclose(new_h @ step, dg, atol=1e-13)
    assert np.min(np.linalg.eigvalsh(new_h)) > 0
    unfavorable = np.array([-0.3, -0.7])
    fallback = bfgs_hessian(h, step, unfavorable)
    np.testing.assert_allclose(fallback @ step, unfavorable, atol=1e-13)
    np.testing.assert_allclose(fallback, fallback.T, atol=1e-13)

    data = make_instanton()
    before = data.G.copy()
    local_step = np.zeros_like(data.x)
    local_step[:, 0, 0] = np.linspace(-0.03, 0.04, len(local_step))
    opt = BandedStreamBedWalk(order=0, executor=QuarticExecutor(), project=False)
    opt.move(data, local_step)
    np.testing.assert_allclose(data.H @ local_step.ravel(), (data.G - before).ravel(), atol=1e-11)


def test_sbw_banded_two_iterations_keep_per_bead_hessians():
    data = make_instanton()
    opt = BandedStreamBedWalk(order=1, executor=QuarticExecutor(), project=False, maxstep=0.3)
    for _ in range(2):
        opt.iterate(data)
        assert data.hess.shape == (8, 3, 3)
        assert np.all(np.isfinite(data.hess))
        np.testing.assert_allclose(
            data.H, unpack_lower_bands(half_ring_bands(data.hess, data.masses, data.springs.omega_n, 3))
        )


def test_sbw_banded_explicit_projection_falls_back_to_dense_eigenpairs():
    data = make_instanton()
    data.n_zero = 3
    opt = BandedStreamBedWalk(order=1, executor=QuarticExecutor(), project=True)
    eigenvalues, eigenvectors = opt._eigenpairs(data)
    assert eigenvalues.shape == (data.x.size - data.n_zero,)
    assert eigenvectors.shape == (data.x.size, data.x.size - data.n_zero)


def test_resolvent_step_matches_all_eigenvectors_on_same_hessian():
    data = make_instanton()
    data.n_zero = 3
    bands = half_ring_bands(data.hess, data.masses, data.springs.omega_n, 3)
    dense = BandedStreamBedWalk(order=1, executor=QuarticExecutor(), project=False, maxstep=0.3)
    values, vectors = dense._eigenpairs(data)
    dense_step = vectors @ dense.scale(dense.step(vectors.T @ data.G.ravel(), values.copy()))
    resolved, _ = banded_sbw_step(bands, data.G.ravel(), data.n_zero, dense.alpha_lambda)
    np.testing.assert_allclose(dense.scale(resolved), dense_step, rtol=1e-9, atol=1e-11)


def test_zero_mode_shift_stays_inside_a_near_zero_cluster():
    values = np.array([-0.057, 0.0, 1e-15, 2e-15, 6.9e-7, 3.9e-6, 3.7e-5, 0.0017])
    shift = _zero_mode_shift(values, np.arange(1, 7))
    np.testing.assert_allclose(shift, 0.5 * values[6])
    np.testing.assert_array_equal(np.sort(np.argsort(abs(values - shift))[:6]), np.arange(1, 7))


def test_single_excluded_mode_uses_banded_step_instead_of_dense_fallback():
    values = np.array([-2.0, 0.0, 1.0, 3.0, 4.0])
    gradient = np.array([0.3, -0.2, 0.5, -0.1, 0.4])
    shift = _zero_mode_shift(values, np.array([1]))
    assert -1.0 < shift < 0.5
    assert abs(shift) > 1e-10
    step, _ = banded_sbw_step(np.array([values]), gradient, 1, StreamBedWalk.alpha_lambda)
    kept = np.array([0, 2, 3, 4])
    alpha, lam = StreamBedWalk.alpha_lambda(values[kept[0]], values[kept[1]])
    expected = np.zeros_like(gradient)
    expected[kept] = alpha * gradient[kept] / (lam - values[kept])
    np.testing.assert_allclose(step, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("values", [np.array([-3.0, 2.0, 4.0]), np.array([1.0, 2.0, 4.0]), np.array([-3.0, -1.0, 2.0])])
def test_resolvent_matches_sbw_across_saddle_spectrum_branches(values):
    gradient = np.array([0.3, -0.2, 0.5])
    step, _ = banded_sbw_step(np.array([values]), gradient, 0, StreamBedWalk.alpha_lambda)
    alpha, lam = StreamBedWalk.alpha_lambda(values[0], values[1])
    np.testing.assert_allclose(step, alpha * gradient / (lam - values), rtol=1e-12, atol=1e-12)


def test_resolvent_optimizer_matches_local_dense_progress():
    reference = make_instanton()
    candidate = make_instanton()
    reference.n_zero = candidate.n_zero = 3
    dense = BandedStreamBedWalk(order=1, executor=QuarticExecutor(), project=False, maxstep=0.3)
    resolved = BandedResolventStreamBedWalk(order=1, executor=QuarticExecutor(), project=False, maxstep=0.3)
    for _ in range(2):
        dense.iterate(reference)
        resolved.iterate(candidate)
        np.testing.assert_allclose(candidate.x, reference.x, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(candidate.G, reference.G, rtol=1e-8, atol=1e-10)
    assert resolved.resolvent_fallbacks == 0


def test_resolvent_search_does_not_assemble_dense_hessian_for_disabled_debug(monkeypatch):
    data = make_instanton()
    data.n_zero = 0
    original = Instanton.H

    def forbid_dense_hessian(self):
        raise AssertionError("ordinary structured search assembled a dense Hessian")

    monkeypatch.setattr(Instanton, "H", property(forbid_dense_hessian, original.fset))
    optimizer = BandedResolventStreamBedWalk(order=1, executor=QuarticExecutor(), project=False, maxstep=0.3)
    optimizer.search(data, gtol=0, maxiter=1)
    assert optimizer.resolvent_fallbacks == 0


def test_resolvent_numerical_failure_falls_back_to_dense(monkeypatch):
    reference = make_instanton()
    candidate = make_instanton()
    reference.n_zero = candidate.n_zero = 3
    dense = BandedStreamBedWalk(order=1, executor=QuarticExecutor(), project=False, maxstep=0.3)
    resolved = BandedResolventStreamBedWalk(order=1, executor=QuarticExecutor(), project=False, maxstep=0.3)

    def fail(*args):
        raise ResolventError("test failure")

    monkeypatch.setattr("pyrinst.opt.optimizers.banded_sbw_step", fail)
    dense.iterate(reference)
    resolved.iterate(candidate)
    np.testing.assert_allclose(candidate.x, reference.x, rtol=1e-12, atol=1e-12)
    assert resolved.resolvent_fallbacks == 1
    assert resolved.resolvent_failure_reasons == ["test failure"]


def test_projection_basis_and_dense_projection_do_not_change_coordinates():
    data = make_projection_instanton()
    original = data.x.copy()
    for metric in ("cartesian", "legacy_mass"):
        constraints = rigid_body_basis(data.x, data.n_zero, data.masses, metric)
        np.testing.assert_allclose(constraints.T @ constraints, np.eye(6), atol=1e-12)
        np.testing.assert_array_equal(data.x, original)
    rot(data.x)
    np.testing.assert_array_equal(data.x, original)
    proj_eig(data.x, data.H.copy(), data.n_zero, mass=data.masses)
    np.testing.assert_array_equal(data.x, original)


def test_projected_resolvent_matches_dense_step_for_both_projection_metrics():
    data = make_projection_instanton()
    bands = half_ring_bands(data.hess, data.masses, data.springs.omega_n, 3)
    optimizer = BandedResolventStreamBedWalk(order=1, executor=ProjectionExecutor(), project=True, maxstep=0.3)
    for metric in ("cartesian", "legacy_mass"):
        constraints = rigid_body_basis(data.x, data.n_zero, data.masses, metric)
        projector = np.eye(data.x.size) - constraints @ constraints.T
        values, vectors = linalg.eigh(projector @ data.H @ projector)
        keep = np.sort(np.argpartition(abs(values), data.n_zero)[data.n_zero :])
        dense_step = vectors[:, keep] @ optimizer.step(vectors[:, keep].T @ data.G.ravel(), values[keep].copy())
        resolved_step, _ = banded_projected_sbw_step(bands, data.G.ravel(), constraints, optimizer.alpha_lambda)
        np.testing.assert_allclose(resolved_step, dense_step, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(constraints.T @ resolved_step, 0, atol=1e-10)


def test_projected_resolvent_uses_selected_modes_without_all_banded_eigenvalues(monkeypatch):
    data = make_projection_instanton()
    bands = half_ring_bands(data.hess, data.masses, data.springs.omega_n, 3)
    constraints = rigid_body_basis(data.x, data.n_zero, data.masses)
    lower_bound = 2 * float(np.min(np.linalg.eigvalsh(data.hess)))

    def reject_all_eigenvalues(*args, **kwargs):
        raise AssertionError("projected SBW requested all banded eigenvalues")

    monkeypatch.setattr(linalg, "eig_banded", reject_all_eigenvalues)
    step, timings = banded_projected_sbw_step(
        bands, data.G.ravel(), constraints, StreamBedWalk.alpha_lambda, potential_lower_bound=lower_bound
    )
    assert np.all(np.isfinite(step))
    assert timings["minimum_eigen_seconds"] > 0
    np.testing.assert_allclose(constraints.T @ step, 0, atol=1e-10)


@pytest.mark.parametrize(
    "shift_selection,preconditioner,band_lu",
    [
        ("eigenvalue", "constrained", False),
        ("bound", "projected", False),
        ("eigenvalue", "projected", True),
        ("bound", "constrained", True),
    ],
)
def test_projected_solver_variants_match_dense_step(shift_selection, preconditioner, band_lu):
    data = make_projection_instanton()
    bands = half_ring_bands(data.hess, data.masses, data.springs.omega_n, 3)
    constraints = rigid_body_basis(data.x, data.n_zero, data.masses)
    projector = np.eye(data.x.size) - constraints @ constraints.T
    values, vectors = linalg.eigh(projector @ data.H @ projector)
    keep = np.sort(np.argpartition(abs(values), data.n_zero)[data.n_zero :])
    dense_step = vectors[:, keep] @ StreamBedWalk.step(
        StreamBedWalk(order=1, executor=ProjectionExecutor()),
        vectors[:, keep].T @ data.G.ravel(), values[keep].copy()
    )
    diagnostics = {}
    step, _ = banded_projected_sbw_step(
        bands, data.G.ravel(), constraints, StreamBedWalk.alpha_lambda,
        shift_selection=shift_selection, preconditioner=preconditioner,
        band_lu=band_lu, diagnostics=diagnostics,
    )
    np.testing.assert_allclose(step, dense_step, rtol=1e-8, atol=1e-10)
    warm_step, _ = banded_projected_sbw_step(
        bands, data.G.ravel(), constraints, StreamBedWalk.alpha_lambda,
        shift_selection=shift_selection, preconditioner=preconditioner,
        band_lu=band_lu, initial_vectors=diagnostics["low_vectors"],
        initial_shift_vector=diagnostics["shift_vector"],
    )
    np.testing.assert_allclose(warm_step, dense_step, rtol=1e-8, atol=1e-10)


def test_shifted_band_lu_matches_dense_indefinite_solve():
    data = make_projection_instanton()
    bands = half_ring_bands(data.hess, data.masses, data.springs.omega_n, 3)
    values = np.linalg.eigvalsh(data.H)
    shift = float((values[1] + values[2]) / 2)
    assert values[0] < shift < values[-1]
    solve = _shifted_band_lu(bands, shift)
    rhs = np.random.default_rng(83).normal(size=(data.x.size, 4))
    reference = np.linalg.solve(data.H - shift * np.eye(data.x.size), rhs)
    np.testing.assert_allclose(solve(rhs), reference, rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(solve(rhs[:, 0]), reference[:, 0], rtol=1e-10, atol=1e-11)


def test_warm_shift_check_rejects_a_missed_near_singular_mode(monkeypatch):
    bands = np.array([[1e-5, 2.0, 4.0, 6.0], [0.0, 0.0, 0.0, 0.0]])
    matrix = _sparse_from_lower_bands(bands)
    wrong_vector = np.array([0.0, 1.0, 0.0, 0.0])

    def stale_eigenpair(*args, **kwargs):
        return np.array([2.0]), wrong_vector[:, None]

    monkeypatch.setattr("pyrinst.opt.sbw_resolvent.eigsh", stale_eigenpair)
    with pytest.raises(ResolventError, match="missed a closer eigenvalue"):
        _nearest_band_eigenvalue(matrix, 0.0, 6.0, bands, initial_vector=wrong_vector)


def test_projected_resolvent_optimizer_matches_corrected_legacy_dense_progress():
    reference = make_projection_instanton()
    candidate = make_projection_instanton()
    dense = BandedStreamBedWalk(order=1, executor=ProjectionExecutor(), project=True, maxstep=0.3)
    resolved = BandedResolventStreamBedWalk(
        order=1, executor=ProjectionExecutor(), project=True, maxstep=0.3, projection_metric="legacy_mass"
    )
    dense.iterate(reference)
    resolved.iterate(candidate)
    np.testing.assert_allclose(candidate.x, reference.x, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(candidate.G, reference.G, rtol=1e-8, atol=1e-10)
    assert resolved.resolvent_fallbacks == 0


def test_projected_optimizer_reuses_low_modes_on_next_iteration(monkeypatch):
    data = make_projection_instanton()
    optimizer = BandedResolventStreamBedWalk(
        order=1, executor=ProjectionExecutor(), project=True, maxstep=0.3,
        band_lu=True, warm_start=True,
    )
    original = banded_projected_sbw_step
    initial_vectors = []
    initial_shift_vectors = []

    def tracked(*args, **kwargs):
        initial_vectors.append(kwargs.get("initial_vectors"))
        initial_shift_vectors.append(kwargs.get("initial_shift_vector"))
        return original(*args, **kwargs)

    monkeypatch.setattr("pyrinst.opt.optimizers.banded_projected_sbw_step", tracked)
    optimizer.iterate(data)
    optimizer.iterate(data)
    assert initial_vectors[0] is None
    assert initial_vectors[1].shape == (data.x.size, 2)
    assert initial_shift_vectors[0] is None
    assert initial_shift_vectors[1].shape == (data.x.size,)
    assert optimizer.resolvent_fallbacks == 0


def test_projected_optimizer_retries_cold_modes_before_dense_fallback(monkeypatch):
    data = make_projection_instanton()
    optimizer = BandedResolventStreamBedWalk(
        order=1, executor=ProjectionExecutor(), project=True, maxstep=0.3,
        band_lu=True, warm_start=True,
    )
    original = banded_projected_sbw_step
    attempts = []

    def fail_warm(*args, **kwargs):
        warm = kwargs.get("initial_vectors") is not None
        attempts.append(warm)
        if warm:
            raise ResolventError("synthetic warm-start failure")
        return original(*args, **kwargs)

    monkeypatch.setattr("pyrinst.opt.optimizers.banded_projected_sbw_step", fail_warm)
    optimizer.iterate(data)
    optimizer.iterate(data)
    assert attempts == [False, True, False]
    assert optimizer.warm_start_retries == 1
    assert optimizer.resolvent_fallbacks == 0


def test_projected_resolvent_fallback_uses_the_same_cartesian_constraints(monkeypatch):
    data = make_projection_instanton()
    initial = data.x.copy()
    constraints = rigid_body_basis(data.x, data.n_zero, data.masses)
    projector = np.eye(data.x.size) - constraints @ constraints.T
    values, vectors = linalg.eigh(projector @ data.H @ projector)
    keep = np.sort(np.argpartition(abs(values), data.n_zero)[data.n_zero :])
    optimizer = BandedResolventStreamBedWalk(order=1, executor=ProjectionExecutor(), project=True, maxstep=0.3)
    dense_step = vectors[:, keep] @ optimizer.scale(
        optimizer.step(vectors[:, keep].T @ data.G.ravel(), values[keep].copy())
    )

    def fail(*args, **kwargs):
        raise ResolventError("test projected failure")

    monkeypatch.setattr("pyrinst.opt.optimizers.banded_projected_sbw_step", fail)
    optimizer.iterate(data)
    np.testing.assert_allclose(data.x, initial + dense_step.reshape(initial.shape), rtol=1e-11, atol=1e-11)
    assert optimizer.resolvent_fallbacks == 1


def test_resolvent_displacement_is_capped_and_checked_before_move(monkeypatch):
    data = make_projection_instanton()
    initial = data.x.copy()
    optimizer = BandedResolventStreamBedWalk(
        order=1, executor=ProjectionExecutor(), project=True, maxstep=0.03
    )
    constraints = rigid_body_basis(data.x, data.n_zero, data.masses)
    candidate = np.random.default_rng(17).normal(size=data.x.size)
    candidate -= constraints @ (constraints.T @ candidate)
    checked = optimizer._checked_displacement(candidate, constraints)
    np.testing.assert_allclose(np.linalg.norm(checked), optimizer.maxstep, rtol=1e-12)
    np.testing.assert_allclose(constraints.T @ checked, 0, atol=1e-12)

    def invalid_step(*args, **kwargs):
        return np.full(data.x.size, np.nan), {}

    monkeypatch.setattr("pyrinst.opt.optimizers.banded_projected_sbw_step", invalid_step)
    optimizer.iterate(data)
    assert optimizer.resolvent_fallbacks == 1
    assert np.all(np.isfinite(data.x))
    assert np.linalg.norm(data.x - initial) <= optimizer.maxstep * (1 + 1e-12)


def test_local_update_skips_near_stationary_bead_with_gradient_noise():
    hess = np.tile(np.diag([-1.0, 1.3, 1.7]), (8, 1, 1))
    step = np.full((8, 3), 0.01)
    step[0] = 1e-10
    step[1] = 1e-7
    grad_change = np.einsum("bij,bj->bi", hess, step)
    grad_change[0, 0] += 1e-8
    grad_change[1, 0] += 1e-5
    updated = update_bead_hessians(hess, step, grad_change, method="powell")
    np.testing.assert_array_equal(updated[0], hess[0])
    np.testing.assert_array_equal(updated[1], hess[1])
    assert np.all(np.isfinite(updated))
