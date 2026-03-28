import numpy as np
import pytest
from unittest.mock import patch, mock_open

from pyrinst.potentials.base import Task
from pyrinst.potentials.orca import Orca, OrcaResult


@pytest.fixture
def mock_engrad_content():
    return """
# Number of atoms
2
# Energy in Hartree
-76.0
# Gradient in Hartree/Bohr
 0.1
 0.2
 0.3
 0.4
 0.5
 0.6
# Atomic numbers and Cartesian Coordinates in Bohr
 8  0.0  0.0  0.0
 1  0.0  0.0  1.0
"""


@pytest.fixture
def mock_hess_content():
    return """
$atoms
2
 0 15.999
 1 1.008
$hessian
6
        0           1           2           3           4
   0    1.000000    2.000000    3.000000    4.000000    5.000000
   1    2.000000    1.000000    2.000000    3.000000    4.000000
   2    3.000000    2.000000    1.000000    2.000000    3.000000
   3    4.000000    3.000000    2.000000    1.000000    2.000000
   4    5.000000    4.000000    3.000000    2.000000    1.000000
   5    6.000000    5.000000    4.000000    3.000000    2.000000
        5
   0    6.000000
   1    5.000000
   2    4.000000
   3    3.000000
   4    2.000000
   5    1.000000
"""


def test_orca_result(tmp_path, mock_engrad_content, mock_hess_content):
    prefix = tmp_path / "test"
    engrad_file = tmp_path / "test.engrad"
    engrad_file.write_text(mock_engrad_content)

    hess_file = tmp_path / "test.hess"
    hess_file.write_text(mock_hess_content)

    res = OrcaResult(str(prefix))

    assert res.symbols == ["O", "H"]
    assert res.energy == -76.0

    expected_coord = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(res.coord, expected_coord)

    expected_grad = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    np.testing.assert_allclose(res.grad, expected_grad)

    assert res.hess.shape == (6, 6)
    assert res.hess[0, 0] == 1.0
    assert res.hess[0, 5] == 6.0

    expected_mass = np.array([15.999, 1.008]) * 1822.888486209 # amu to au approximately
    # the units conversion scales mass by Mass(1, "amu").get("au")
    # we just check shape and nonzero
    assert res.mass.shape == (2,)
    assert res.mass[0] > 0
    assert res.mass[1] > 0


@pytest.fixture
def mock_orca_template(tmp_path):
    template = tmp_path / "template.inp"
    template.write_text("! UKS B3LYP Def2-SVP\n%maxcore 2000\n* xyz 0 1\n")
    return str(template)


def test_orca_generate_input(tmp_path, mock_orca_template):
    drv = Orca(symbols=["H", "H"], template_input=mock_orca_template, working_dir=str(tmp_path))
    drv._sys_name = "test"
    drv._folder = str(tmp_path)

    x = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    drv.generate_input(x, Task.GRAD)

    infile = tmp_path / "test.inp"
    assert infile.exists()

    content = infile.read_text()
    assert "! UKS B3LYP Def2-SVP" in content
    assert "! EnGrad" in content
    assert "* xyz 0 1" in content
    assert "H" in content


@patch("subprocess.run")
@patch("pyrinst.potentials.orca.OrcaResult")
def test_orca_call(mock_result, mock_subprocess, tmp_path, mock_orca_template):
    drv = Orca(symbols=["H", "H"], template_input=mock_orca_template, working_dir=str(tmp_path))
    x = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

    res_inst = mock_result.return_value
    res_inst.energy = -1.0
    res_inst.grad = np.zeros((2, 3))
    res_inst.hess = np.zeros((6, 6))

    e, g, h = drv(x, Task.GRAD)

    assert mock_subprocess.called
    assert mock_result.called
    assert e == -1.0
    np.testing.assert_array_equal(g, np.zeros((2, 3)))
