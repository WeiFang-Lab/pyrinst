"""Exact banded spectra for Pyrinst's full-ring instanton Hessian."""

from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pyrinst.geometries import Instanton, InstRef
from pyrinst.thermo import ThermoData
from pyrinst.utils.coordinates import mass_weight


def make_instanton(n_beads: int, full_ring: bool) -> Instanton:
    rng = np.random.default_rng(100 + n_beads)
    n_stored = n_beads if full_ring else n_beads // 2
    inst = Instanton(
        rng.normal(size=(n_stored, 2, 3)),
        masses=np.array([2.0, 5.0]),
        beta=100.0,
        full_ring=full_ring,
    )
    blocks = rng.normal(size=(n_stored, inst.dof, inst.dof))
    inst.hess = (blocks + blocks.transpose(0, 2, 1)) / 2
    return inst


def unpack_lower(band: np.ndarray) -> np.ndarray:
    n = band.shape[1]
    dense = np.zeros((n, n))
    for col in range(n):
        count = min(len(band), n - col)
        dense[col : col + count, col] = band[:count, col]
    return np.tril(dense) + np.tril(dense, -1).T


@pytest.mark.parametrize(
    "n_beads,full_ring", [(2, True), (3, True), (4, True), (8, True), (2, False), (4, False), (8, False)]
)
def test_banded_hessian_preserves_full_ring_spectrum(n_beads: int, full_ring: bool) -> None:
    inst = make_instanton(n_beads, full_ring)
    band = inst.hessian_full_banded()
    dense = mass_weight(inst.hessian_full(), inst.m, dim=3)
    order = [0]
    left, right = 1, n_beads - 1
    while left <= right:
        order.append(left)
        if left < right:
            order.append(right)
        left += 1
        right -= 1
    perm = (np.array(order)[:, None] * inst.dof + np.arange(inst.dof)).ravel()
    assert_allclose(unpack_lower(band), dense[np.ix_(perm, perm)], rtol=1e-13, atol=1e-13)
    assert_allclose(np.linalg.eigvalsh(unpack_lower(band)), np.linalg.eigvalsh(dense), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("full_ring", [False, True])
def test_instanton_vib_uses_banded_solver(full_ring: bool) -> None:
    inst = make_instanton(8, full_ring)
    inst.set_beta(10.0)
    inst.n_zero = 0
    inst.hess[:] = np.eye(inst.dof)
    inst.hess[0, 0, 0] = -2
    inst.hess[:, 1, 1] = 0
    dense_eigenvalues = np.linalg.eigvalsh(mass_weight(inst.hessian_full(), inst.m, dim=3))
    with patch.object(Instanton, "hessian_full", side_effect=AssertionError("dense Hessian assembled")):
        data = ThermoData(inst.beta, "inst")
        inst.vib(data, inst.beta)
    expected_freqs = np.sqrt(abs(dense_eigenvalues)) * np.sign(dense_eigenvalues)
    expected_nonzero = np.sort(expected_freqs[np.argsort(abs(expected_freqs))[1:]])
    assert_allclose(data.freqs, expected_nonzero, atol=1e-12)
    assert np.isfinite(data.log_pf[2])


def test_instref_keeps_projected_dense_path() -> None:
    inst = InstRef(np.array([[[0.0], [0.0]], [[0.1], [0.2]]]), masses=np.array([1.0, 2.0]), beta=100)
    inst.hess = np.tile(np.eye(inst.dof), (len(inst.x), 1, 1))
    with pytest.raises(NotImplementedError, match="Projected"):
        inst.hessian_full_banded()
    projected = np.diag([-1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    with (
        patch("pyrinst.geometries.eig_banded", side_effect=AssertionError("banded solver called")),
        patch.object(InstRef, "hessian_full", return_value=projected),
    ):
        data = ThermoData(inst.beta, "centroid")
        inst.vib(data, inst.beta)
    assert np.isfinite(data.log_pf[2])


def test_stored_dense_full_ring_hessian_keeps_dense_path() -> None:
    inst = Instanton(
        np.array([[[0.0], [0.0]], [[0.1], [0.2]], [[0.2], [0.3]], [[0.3], [0.4]]]),
        masses=np.array([1.0, 2.0]),
        beta=100,
        full_ring=True,
    )
    inst.hess = np.diag([-2.0, 0.001, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    with patch("pyrinst.geometries.eig_banded", side_effect=AssertionError("banded solver called")):
        data = ThermoData(inst.beta, "inst")
        inst.vib(data, inst.beta)
    assert np.isfinite(data.log_pf[2])
