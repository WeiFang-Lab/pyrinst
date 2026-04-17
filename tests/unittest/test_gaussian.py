import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from pyrinst.potentials.base import Task
from pyrinst.potentials.gaussian import Gaussian, GaussianResult
from pyrinst.utils.units import Mass


@pytest.fixture
def tmp_path():
    p = Path.cwd() / "tmp_test"
    p.mkdir(exist_ok=True, parents=True)
    yield p
    shutil.rmtree(p)


@pytest.fixture
def mock_fchk_content():
    return """Title
Some other line
Atomic numbers             I    N=           3
           8           1           1
Current cartesian coordinates   R   N=           9
  0.0 0.0 0.0 1.0 1.0
  1.0 -1.0 -1.0 -1.0
Real atomic weights        R    N=           3
  15.999 1.008 1.008
Total Energy               R                 -76.0
Cartesian Gradient         R    N=           9
  0.1 0.2 0.3 0.4 0.5
  0.6 0.7 0.8 0.9
Cartesian Force Constants  R    N=          45
  1.0 2.0 3.0 4.0 5.0
  6.0 7.0 8.0 9.0 10.0
  11.0 12.0 13.0 14.0 15.0
  16.0 17.0 18.0 19.0 20.0
  21.0 22.0 23.0 24.0 25.0
  26.0 27.0 28.0 29.0 30.0
  31.0 32.0 33.0 34.0 35.0
  36.0 37.0 38.0 39.0 40.0
  41.0 42.0 43.0 44.0 45.0
"""


def test_gaussian_result(tmp_path, mock_fchk_content):
    prefix = tmp_path / "test"
    fchk_file = tmp_path / "test.fchk"
    fchk_file.write_text(mock_fchk_content)

    res = GaussianResult(str(prefix))
    assert (res.symbols == ["O", "H", "H"]).all()
    assert res.energy == -76.0
    np.testing.assert_allclose(res.coord, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])

    expected_mass = np.array([15.999, 1.008, 1.008]) * Mass(1, "amu").get("au")
    np.testing.assert_allclose(res.mass, expected_mass)

    expected_grad = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).reshape(3, 3)
    np.testing.assert_allclose(res.grad, expected_grad)

    # Check that hessian is 9x9 symmetric
    assert res.hess.shape == (9, 9)
    assert np.allclose(res.hess, res.hess.T)
    # the diagonal elements (1st, 3rd, 6th... element from the array reading form)
    # The first element is 1.0
    assert res.hess[0, 0] == 1.0


@pytest.fixture
def mock_gaussian_template():
    return str(Path(__file__).parent / "template_gaussian.txt")


def test_gaussian_generate_input(tmp_path, mock_gaussian_template):
    drv = Gaussian(symbols=["H", "H"], template_input=mock_gaussian_template, working_dir=str(tmp_path))
    drv._sys_name = "test"
    drv._input = "test.com"
    drv._folder = str(tmp_path)

    x = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    drv.generate_input(x, Task.GRAD)

    infile = tmp_path / "test.com"
    assert infile.exists()

    content = infile.read_text()
    assert "Force" in content
    assert "0 1" in content
    assert "H" in content


@patch("subprocess.run")
@patch("pyrinst.potentials.gaussian.GaussianResult")
def test_gaussian_call(mock_result, mock_subprocess, tmp_path, mock_gaussian_template):
    drv = Gaussian(symbols=["H", "H"], template_input=mock_gaussian_template, working_dir=str(tmp_path))
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
