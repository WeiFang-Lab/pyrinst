"""Physical identities for the bead-local SBW Hessian update."""

import numpy as np
import pytest

from pyrinst.geometries import Instanton
from pyrinst.opt.hessian import project_rigid_hessian, rigid_hessian_directions, update_bead_hessians_rigid
from pyrinst.opt.optimizers import OPTIMIZER_REGISTRY, RigidBandedResolventStreamBedWalk
from pyrinst.potentials import Level


def radial_pair(coords):
    """Analytic translation- and rotation-invariant V=|q1-q0|^4/4."""
    separation = coords[1] - coords[0]
    radius_squared = separation @ separation
    gradient = np.stack((-radius_squared * separation, radius_squared * separation))
    curvature = radius_squared * np.eye(3) + 2 * np.outer(separation, separation)
    hessian = np.block([[curvature, -curvature], [-curvature, curvature]])
    return radius_squared**2 / 4, gradient, hessian


def quartic_pairs(coords):
    """Analytic quartic pair potential for a noncollinear molecule."""
    atoms = len(coords)
    gradient = np.zeros_like(coords)
    hessian = np.zeros((3 * atoms, 3 * atoms))
    for first in range(atoms):
        for second in range(first + 1, atoms):
            separation = coords[second] - coords[first]
            radius_squared = separation @ separation
            force = radius_squared * separation
            curvature = radius_squared * np.eye(3) + 2 * np.outer(separation, separation)
            gradient[first] -= force
            gradient[second] += force
            one = slice(3 * first, 3 * first + 3)
            two = slice(3 * second, 3 * second + 3)
            hessian[one, one] += curvature
            hessian[two, two] += curvature
            hessian[one, two] -= curvature
            hessian[two, one] -= curvature
    return gradient, hessian


class RadialPairExecutor:
    def compute(self, data, level=Level.GRAD):
        values = [radial_pair(bead) for bead in data.x]
        data.V = np.array([value[0] for value in values])
        data.G = np.array([value[1] for value in values])
        data.H = np.array([value[2] for value in values]) if level == Level.FREQ else None


def test_rigid_update_preserves_endpoint_identities_and_best_compatible_secant():
    old = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    new = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    _, old_gradient, old_hessian = radial_pair(old)
    _, new_gradient, exact_hessian = radial_pair(new)
    step = (new - old).ravel()
    difference = (new_gradient - old_gradient).ravel()
    updated = update_bead_hessians_rigid(
        old_hessian[None], new[None], new_gradient[None], (new - old)[None], (new_gradient - old_gradient)[None]
    )[0]
    directions, images = rigid_hessian_directions(new, new_gradient)
    np.testing.assert_allclose(updated, updated.T, atol=1e-13)
    np.testing.assert_allclose(updated @ directions, images, atol=1e-12)
    np.testing.assert_allclose(exact_hessian @ directions, images, atol=1e-12)

    rigid_basis, singular, _ = np.linalg.svd(directions, full_matrices=False)
    rigid_basis = rigid_basis[:, : np.count_nonzero(singular > 1e-10 * singular[0])]
    free_projector = np.eye(len(step)) - rigid_basis @ rigid_basis.T
    np.testing.assert_allclose(free_projector @ (updated @ step - difference), 0, atol=1e-12)
    assert np.linalg.norm(updated @ step - difference) > 0.01


def test_rigid_projection_rejects_noninvariant_gradient():
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    _, gradient, hessian = radial_pair(coords)
    gradient[0, 0] += 0.1
    with pytest.raises(ValueError, match="incompatible|no symmetric"):
        project_rigid_hessian(hessian, coords, gradient)
    with pytest.raises(ValueError, match="incompatible|no symmetric"):
        project_rigid_hessian(hessian * 1e9, coords, gradient)


def test_batched_rigid_update_keeps_covariance_on_noncollinear_beads():
    old = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 0.9, 0.1]],
            [[0.1, 0.0, 0.0], [1.1, 0.1, 0.0], [0.2, 0.9, 0.2]],
            [[0.0, 0.1, 0.1], [0.9, 0.1, 0.0], [0.3, 1.0, 0.1]],
        ]
    )
    steps = np.array(
        [
            [[0.01, -0.02, 0.0], [0.0, 0.03, 0.01], [-0.01, 0.0, 0.02]],
            [[0.02, 0.01, 0.0], [0.0, -0.01, 0.03], [0.01, 0.02, 0.0]],
            [[0.0, -0.01, 0.02], [0.02, 0.0, 0.0], [0.01, 0.01, -0.01]],
        ]
    )
    new = old + steps
    old_values = [quartic_pairs(bead) for bead in old]
    new_values = [quartic_pairs(bead) for bead in new]
    old_hessians = np.array([item[1] for item in old_values])
    gradients = np.array([item[0] for item in new_values])
    changes = gradients - np.array([item[0] for item in old_values])
    updated = update_bead_hessians_rigid(old_hessians, new, gradients, steps, changes)
    for hessian, bead, gradient, step, change in zip(updated, new, gradients, steps, changes, strict=True):
        directions, images = rigid_hessian_directions(bead, gradient)
        np.testing.assert_allclose(hessian, hessian.T, atol=1e-12)
        np.testing.assert_allclose(hessian @ directions, images, atol=1e-11)
        basis, _, _ = np.linalg.svd(directions, full_matrices=False)
        projector = np.eye(len(hessian)) - basis @ basis.T
        np.testing.assert_allclose(projector @ (hessian @ step.ravel() - change.ravel()), 0, atol=1e-11)

    angle = 0.37
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    transform = np.kron(np.eye(3), rotation)
    translated = new @ rotation.T + np.array([2.0, -1.0, 0.5])
    transformed_hessians = transform @ old_hessians @ transform.T
    transformed = update_bead_hessians_rigid(
        transformed_hessians,
        translated,
        gradients @ rotation.T,
        steps @ rotation.T,
        changes @ rotation.T,
    )
    np.testing.assert_allclose(transformed, transform @ updated @ transform.T, atol=1e-11)


def test_rigid_resolvent_optimizer_uses_physical_local_update():
    assert OPTIMIZER_REGISTRY["SBWRP"] is RigidBandedResolventStreamBedWalk
    old = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    coords = np.tile(old, (4, 1, 1))
    data = Instanton(coords, ["H", "H"], masses=np.ones(2), beta=25.0, n_zero=5)
    executor = RadialPairExecutor()
    executor.compute(data, level=Level.FREQ)
    displacement = np.zeros_like(coords)
    displacement[:, 1, 1] = 0.2
    optimizer = RigidBandedResolventStreamBedWalk(order=1, executor=executor, project=True)
    optimizer.move(data, displacement)
    for hessian, bead, gradient in zip(data.hess, data.x, data.grad, strict=True):
        directions, images = rigid_hessian_directions(bead, gradient)
        np.testing.assert_allclose(hessian @ directions, images, atol=1e-12)
