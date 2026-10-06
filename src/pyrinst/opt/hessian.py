import numpy as np
from numpy.typing import NDArray
from scipy import linalg


def bfgs(hess: NDArray, d: NDArray, dg: NDArray) -> NDArray:
    hy = linalg.blas.dgemv(1.0, hess, d)  # use scipy blas for efficiency
    dy = np.dot(d, dg)
    dhy = np.dot(d, hy)
    if dy <= 0 or dhy <= 0:
        return hess
    for i in range(len(hess)):  # memory efficient implementation without efficiency loss
        hess[i] += dg[i] * dg / dy - hy[i] * hy / dhy
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
    if d2 == 0 or xi2 <= np.finfo(float).eps ** 2 * max(np.dot(dg, dg), np.dot(hess @ d, hess @ d)):
        return hess
    # Algebraically equivalent mixing avoids division by d.xi (which may vanish).
    phi = np.clip(dxi**2 / (d2 * xi2), 0, 1)
    psb = (np.outer(xi, d) + np.outer(d, xi)) / d2 - dxi * np.outer(d, d) / d2**2
    return hess + dxi * np.outer(xi, xi) / (d2 * xi2) + (1 - phi) * psb


def bfgs_hessian(hess: NDArray, d: NDArray, dg: NDArray) -> NDArray:
    """Hessian-form BFGS update, with a symmetric fallback for bad curvature."""
    hd = hess @ d
    dy = np.dot(d, dg)
    dhd = np.dot(d, hd)
    if dy <= 1e-8 * np.linalg.norm(d) * np.linalg.norm(dg) or dhd <= 1e-8 * np.linalg.norm(d) * np.linalg.norm(hd):
        return powell(hess, d, dg)
    return hess + np.outer(dg, dg) / dy - np.outer(hd, hd) / dhd


def update_bead_hessians(
    hess: NDArray, d: NDArray, dg: NDArray, method: str = "bofill", max_update_ratio: float = 5.0
) -> NDArray:
    """Update independent bead potential Hessians without filling spring bands.

    ``dg`` is the change in the *potential* gradient at each bead. The spring
    Hessian is constant and must be added separately when building the ring
    polymer Hessian. ``method`` selects the local EF or SBW update. Powell's
    symmetric update handles a near-zero denominator or failed BFGS curvature
    condition. Beads with negligible displacement are left unchanged, since
    dividing gradient noise by their step would produce a large correction.
    Corrections larger than ``max_update_ratio`` times the current block norm
    are also skipped; this safeguard should be calibrated on real gradients.
    """
    if hess.ndim != 3 or hess.shape[1] != hess.shape[2]:
        raise ValueError("hess must have shape (nbeads, dof, dof)")
    if method not in {"bofill", "powell", "bfgs"}:
        raise ValueError(f"unsupported bead Hessian update: {method}")
    if max_update_ratio <= 0:
        raise ValueError("max_update_ratio must be positive")
    steps = np.asarray(d).reshape(len(hess), -1)
    grad_steps = np.asarray(dg).reshape(len(hess), -1)
    if steps.shape[1] != hess.shape[1] or grad_steps.shape != steps.shape:
        raise ValueError("displacement and gradient change must match bead Hessians")

    updated = hess.copy()
    min_step = max(1e-8, 1e-6 * np.linalg.norm(steps))
    for i, (step, grad_step) in enumerate(zip(steps, grad_steps, strict=True)):
        step_norm = np.linalg.norm(step)
        if step_norm <= min_step:
            continue
        residual = grad_step - updated[i] @ step
        residual_norm = np.linalg.norm(residual)
        if residual_norm <= 1e-12 * max(np.linalg.norm(grad_step), np.linalg.norm(updated[i] @ step)):
            continue
        if method == "powell":
            candidate = powell(updated[i].copy(), step, grad_step)
        elif method == "bfgs":
            candidate = bfgs_hessian(updated[i].copy(), step, grad_step)
        else:
            curvature = np.dot(step, residual)
            if abs(curvature) > 1e-8 * step_norm * residual_norm:
                candidate = bofill(updated[i].copy(), step, grad_step)
            else:
                candidate = powell(updated[i].copy(), step, grad_step)
        if np.all(np.isfinite(candidate)) and np.linalg.norm(candidate - updated[i]) <= max_update_ratio * max(
            1.0, np.linalg.norm(updated[i])
        ):
            updated[i] = candidate
    return updated


def rigid_hessian_directions(coords: NDArray, gradient: NDArray) -> tuple[NDArray, NDArray]:
    """Return the Cartesian rigid directions D and their Hessian images V.

    For a translation- and rotation-invariant potential, its exact Hessian K
    satisfies K @ D = V. The rotational image is J @ gradient, rather than
    zero away from a stationary geometry. Coordinates are centered only to
    improve conditioning; a translation-invariant Hessian annihilates the
    difference between centered and uncentered rotations.
    """
    coords = np.asarray(coords)
    gradient = np.asarray(gradient)
    if coords.ndim != 2 or coords.shape[1] != 3 or gradient.shape != coords.shape:
        raise ValueError("rigid Hessian constraints require matching (atoms, 3) coordinates and gradients")
    atoms = len(coords)
    if atoms < 1:
        raise ValueError("rigid Hessian constraints require at least one atom")
    translations = np.tile(np.eye(3), (atoms, 1)) / np.sqrt(atoms)
    centered = coords - coords.mean(axis=0)
    axes = np.eye(3)
    rotations = np.column_stack([np.cross(axis, centered).reshape(-1) for axis in axes])
    gradient_rotations = np.column_stack([np.cross(axis, gradient).reshape(-1) for axis in axes])
    directions = np.column_stack((translations, rotations))
    images = np.column_stack((np.zeros_like(translations), gradient_rotations))
    return directions, images


def _project_rigid_hessian(
    hess: NDArray, coords: NDArray, gradient: NDArray, tolerance: float
) -> tuple[NDArray, NDArray]:
    """Return the closest physical Hessian and an orthonormal rigid basis.

    Minimize ||K - hess||_F over symmetric K satisfying K @ D = V, where
    ``rigid_hessian_directions`` supplies D and V. Incompatible PES gradients
    are rejected instead of silently changing the required physical images.
    SVD handles the five independent rigid directions of a linear geometry.
    """
    directions, images = rigid_hessian_directions(coords, gradient)
    size = directions.shape[0]
    if hess.shape != (size, size) or tolerance <= 0:
        raise ValueError("invalid rigid Hessian shape or tolerance")
    if not (np.all(np.isfinite(hess)) and np.all(np.isfinite(directions)) and np.all(np.isfinite(images))):
        raise ValueError("nonfinite rigid Hessian inputs")
    left, singular, right_t = np.linalg.svd(directions, full_matrices=False)
    rank = int(np.count_nonzero(singular > 1e-10 * singular[0]))
    basis = left[:, :rank]
    reduced_target = images @ right_t[:rank].T / singular[:rank]
    reconstructed = (reduced_target * singular[:rank]) @ right_t[:rank]
    target_scale = max(1.0, np.linalg.norm(images), np.linalg.norm(reconstructed))
    if np.linalg.norm(reconstructed - images) > tolerance * target_scale:
        raise ValueError("PES gradient is incompatible with rigid directions")
    constrained_block = basis.T @ reduced_target
    if np.linalg.norm(constrained_block - constrained_block.T) > tolerance * max(1.0, np.linalg.norm(reduced_target)):
        raise ValueError("no symmetric Hessian satisfies the PES rigid identities")
    constrained_block = (constrained_block + constrained_block.T) / 2
    projector = np.eye(size) - basis @ basis.T
    cross = projector @ reduced_target
    symmetric_prior = (hess + hess.T) / 2
    projected = (
        projector @ symmetric_prior @ projector
        + cross @ basis.T
        + basis @ cross.T
        + basis @ constrained_block @ basis.T
    )
    projected = (projected + projected.T) / 2
    scale = max(1.0, np.linalg.norm(images), np.linalg.norm(projected) * np.linalg.norm(directions))
    if np.linalg.norm(projected @ directions - images) > tolerance * scale:
        raise ValueError("rigid Hessian projection failed its residual check")
    return projected, basis


def project_rigid_hessian(hess: NDArray, coords: NDArray, gradient: NDArray, tolerance: float = 1e-9) -> NDArray:
    """Project onto symmetric rigid identities with minimum Frobenius change."""
    return _project_rigid_hessian(hess, coords, gradient, tolerance)[0]


def _rigid_update_full_rank(
    hess: NDArray,
    steps: NDArray,
    changes: NDArray,
    directions: NDArray,
    images: NDArray,
    left: NDArray,
    singular: NDArray,
    right_t: NDArray,
    min_step: float,
    max_update_ratio: float,
) -> NDArray:
    """Batch the common six-dimensional rigid case without changing its math."""

    def transpose(matrices: NDArray) -> NDArray:
        return np.swapaxes(matrices, -1, -2)

    basis = left
    targets = (images @ transpose(right_t)) / singular[:, None, :]
    reconstructed = (targets * singular[:, None, :]) @ right_t
    target_scale = np.maximum(
        1.0, np.maximum(np.linalg.norm(images, axis=(1, 2)), np.linalg.norm(reconstructed, axis=(1, 2)))
    )
    if np.any(np.linalg.norm(reconstructed - images, axis=(1, 2)) > 1e-9 * target_scale):
        raise ValueError("PES gradient is incompatible with rigid directions")
    constrained = transpose(basis) @ targets
    if np.any(
        np.linalg.norm(constrained - transpose(constrained), axis=(1, 2))
        > 1e-9 * np.maximum(1.0, np.linalg.norm(targets, axis=(1, 2)))
    ):
        raise ValueError("no symmetric Hessian satisfies the PES rigid identities")
    constrained = (constrained + transpose(constrained)) / 2
    projector = np.eye(hess.shape[-1]) - basis @ transpose(basis)
    cross = projector @ targets
    symmetric_prior = (hess + transpose(hess)) / 2
    physical = (
        projector @ symmetric_prior @ projector
        + cross @ transpose(basis)
        + basis @ transpose(cross)
        + basis @ constrained @ transpose(basis)
    )
    physical = (physical + transpose(physical)) / 2
    scale = np.maximum(
        1.0,
        np.maximum(
            np.linalg.norm(images, axis=(1, 2)),
            np.linalg.norm(physical, axis=(1, 2)) * np.linalg.norm(directions, axis=(1, 2)),
        ),
    )
    if np.any(np.linalg.norm(physical @ directions - images, axis=(1, 2)) > 1e-9 * scale):
        raise ValueError("rigid Hessian projection failed its residual check")

    tangent_step = steps - (basis @ (transpose(basis) @ steps[..., None]))[..., 0]
    error = changes - (physical @ steps[..., None])[..., 0]
    residual = error - (basis @ (transpose(basis) @ error[..., None]))[..., 0]
    length_squared = np.sum(tangent_step**2, axis=1)
    usable = length_squared > min_step**2
    usable &= np.linalg.norm(residual, axis=1) > 1e-12 * np.maximum(
        np.linalg.norm(changes, axis=1), np.linalg.norm((physical @ steps[..., None])[..., 0], axis=1)
    )
    divisor = np.where(usable, length_squared, 1.0)
    correction = (
        residual[..., None] * tangent_step[:, None, :] + tangent_step[..., None] * residual[:, None, :]
    ) / divisor[:, None, None] - np.sum(residual * tangent_step, axis=1)[:, None, None] * (
        tangent_step[..., None] * tangent_step[:, None, :]
    ) / divisor[:, None, None] ** 2
    usable &= np.all(np.isfinite(correction), axis=(1, 2))
    usable &= np.linalg.norm(correction, axis=(1, 2)) <= max_update_ratio * np.maximum(
        1.0, np.linalg.norm(physical, axis=(1, 2))
    )
    updated = physical + np.where(usable[:, None, None], correction, 0.0)
    return (updated + transpose(updated)) / 2


def update_bead_hessians_rigid(
    hess: NDArray,
    coords: NDArray,
    gradient: NDArray,
    d: NDArray,
    dg: NDArray,
    max_update_ratio: float = 5.0,
) -> NDArray:
    """Update independent PES blocks while preserving rigid Hessian identities.

    First project each old block onto the affine physical constraints at the
    *new* coordinates. In the unconstrained subspace, apply the minimum-change
    symmetric Powell update to satisfy the projected secant equation. The
    remaining rigid component of the finite-step secant may be incompatible
    with exact endpoint covariance and is therefore not forced.
    """
    coords = np.asarray(coords)
    gradient = np.asarray(gradient)
    d = np.asarray(d)
    dg = np.asarray(dg)
    if hess.ndim != 3 or hess.shape[0] < 1 or hess.shape[1] != hess.shape[2] or max_update_ratio <= 0:
        raise ValueError("invalid bead Hessians or update limit")
    if coords.ndim != 3 or coords.shape != gradient.shape or coords.shape[-1] != 3 or coords.shape[1] < 1:
        raise ValueError("rigid update requires matching three-dimensional coordinates and gradients")
    beads, atoms, _ = coords.shape
    if hess.shape != (beads, 3 * atoms, 3 * atoms) or d.shape != coords.shape or dg.shape != coords.shape:
        raise ValueError("rigid bead update arrays have inconsistent shapes")
    steps = np.asarray(d).reshape(beads, -1)
    changes = np.asarray(dg).reshape(beads, -1)
    min_step = max(1e-8, 1e-6 * np.linalg.norm(steps))
    axes = np.eye(3)
    translations = np.broadcast_to(np.tile(axes, (atoms, 1)) / np.sqrt(atoms), (beads, 3 * atoms, 3))
    centered = coords - coords.mean(axis=1, keepdims=True)
    rotations = np.stack([np.cross(axis, centered).reshape(beads, -1) for axis in axes], axis=-1)
    gradient_rotations = np.stack([np.cross(axis, gradient).reshape(beads, -1) for axis in axes], axis=-1)
    directions = np.concatenate((translations, rotations), axis=-1)
    images = np.concatenate((np.zeros_like(translations), gradient_rotations), axis=-1)
    if not (np.all(np.isfinite(hess)) and np.all(np.isfinite(directions)) and np.all(np.isfinite(images))):
        raise ValueError("nonfinite rigid Hessian inputs")
    left, singular, right_t = np.linalg.svd(directions, full_matrices=False)
    if singular.shape[1] == 6 and np.all(singular[:, -1] > 1e-10 * singular[:, 0]):
        return _rigid_update_full_rank(
            hess,
            steps,
            changes,
            directions,
            images,
            left,
            singular,
            right_t,
            min_step,
            max_update_ratio,
        )
    updated = np.empty_like(hess)
    for i, (prior, x, grad, step, change) in enumerate(zip(hess, coords, gradient, steps, changes, strict=True)):
        physical, basis = _project_rigid_hessian(prior, x, grad, 1e-9)
        tangent_step = step - basis @ (basis.T @ step)
        length_squared = float(tangent_step @ tangent_step)
        if length_squared <= min_step**2:
            updated[i] = physical
            continue
        error = change - physical @ step
        residual = error - basis @ (basis.T @ error)
        if np.linalg.norm(residual) <= 1e-12 * max(np.linalg.norm(change), np.linalg.norm(physical @ step)):
            updated[i] = physical
            continue
        correction = (np.outer(residual, tangent_step) + np.outer(tangent_step, residual)) / length_squared - (
            residual @ tangent_step
        ) * np.outer(tangent_step, tangent_step) / length_squared**2
        if np.all(np.isfinite(correction)) and np.linalg.norm(correction) <= max_update_ratio * max(
            1.0, np.linalg.norm(physical)
        ):
            physical += correction
        updated[i] = (physical + physical.T) / 2
    return updated


def half_ring_bands(hess: NDArray, masses: NDArray, omega_n: float, dim: int) -> NDArray:
    """Return the lower-band storage of a half-ring instanton Hessian.

    ``hess`` contains the unweighted potential Hessian at each bead. The
    returned array has the format required by ``scipy.linalg.eig_banded``.
    """
    beads, dof, dof2 = hess.shape
    if beads < 2 or dof != dof2 or dof != len(masses) * dim:
        raise ValueError("invalid bead Hessian shape")
    spring_diag = 2 * np.repeat(masses, dim) * omega_n**2
    bands = np.zeros((dof + 1, beads * dof), dtype=hess.dtype)
    for bead in range(beads):
        start = bead * dof
        for col in range(dof):
            bands[0, start + col] = 2 * hess[bead, col, col] + spring_diag[col] * (1 if bead in (0, beads - 1) else 2)
            for row in range(col + 1, dof):
                bands[row - col, start + col] = 2 * hess[bead, row, col]
            if bead < beads - 1:
                bands[dof, start + col] = -spring_diag[col]
    return bands
