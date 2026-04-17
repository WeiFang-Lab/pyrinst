"""
Unit tests for the mechanics module in `pyrinst.utils.mechanics`.
"""

import numpy as np

from pyrinst.utils.mechanics import center_of_mass, inertia


def test_center_of_mass_single_molecule():
    coords = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    com = center_of_mass(coords)
    np.testing.assert_allclose(com, [1 / 3, 1 / 3, 1 / 3])


def test_center_of_mass_water_docstring_example():
    # Coordinates for a water molecule (O, H, H) in Angstroms
    coords = np.array([[0.0000, 0.0000, 0.1173], [0.0000, 0.7572, -0.4692], [0.0000, -0.7572, -0.4692]])
    # Masses in atomic mass units (amu)
    masses = np.array([15.999, 1.008, 1.008])
    expected_com = np.array([0.0, 0.0, 0.05166669])
    com = center_of_mass(coords, masses)
    np.testing.assert_allclose(com, expected_com, atol=1e-7)


def test_center_of_mass_batch():
    coords = np.array([[[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]])
    masses = np.array([1.0, 1.0])
    com = center_of_mass(coords, masses)
    # Based on current function implementation which returns np.mean(res, axis=0) if res.ndim == 2
    # res.shape = (2, 3), so return shape is (3,)
    expected_com = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(com, expected_com)


def test_inertia_water_docstring_example():
    # Coordinates for a water molecule (O, H, H) in Angstroms
    coords = np.array([[0.0000, 0.0000, 0.1173], [0.0000, 0.7572, -0.4692], [0.0000, -0.7572, -0.4692]])
    # Masses in atomic mass units (amu)
    masses = np.array([15.999, 1.008, 1.008])
    moi = inertia(coords, masses)

    expected_moi = np.array([[1.7717, 0.0, 0.0], [0.0, 0.6159, 0.0], [0.0, 0.0, 1.1559]])
    np.testing.assert_allclose(moi, expected_moi, atol=1e-4)

    p_moments = np.linalg.eigvalsh(moi)
    expected_p_moments = np.array([0.6159, 1.1559, 1.7717])
    np.testing.assert_allclose(p_moments, expected_p_moments, atol=1e-4)


def test_inertia_com_false():
    coords = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    moi = inertia(coords, mass=1.0, com=False)
    # m=1, r1=(0,1,0), r2=(0,-1,0). r^2=1
    # diag = r1^2 + r2^2 = 2.
    # outer = [[x^2, xy, xz], [yx, y^2, yz], [zx, zy, z^2]]
    # outer1 = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    # outer2 = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    # outer_sum = [[0, 0, 0], [0, 2, 0], [0, 0, 0]]
    # I = 2*eye - outer_sum = [[2, 0, 0], [0, 0, 0], [0, 0, 2]]
    expected_moi = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    np.testing.assert_allclose(moi, expected_moi)
