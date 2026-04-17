from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pyrinst.potentials.base import Task

pytest.importorskip("ase")
pytest.importorskip("mace.calculators")
from pyrinst.potentials.mace import MACE
from pyrinst.utils.units import Energy, Length


def test_mace_init():
    with patch("pyrinst.potentials.mace.MACECalculator") as mock_calc_class:
        mock_calc_class.return_value = "fake_calculator"
        mace = MACE(symbols=["H", "H"], model_paths="fake_path", device="cpu")

        assert mace.symbols == ["H", "H"]
        assert len(mace.atoms) == 2
        assert mace.atoms.calc == "fake_calculator"
        mock_calc_class.assert_called_once_with(
            model_paths="fake_path",
            default_dtype="float64",
            device="cpu",
            enable_cueq=False,
        )


def test_mace_call():
    with patch("pyrinst.potentials.mace.MACECalculator") as mock_calc_class:
        mock_calc = MagicMock()
        mock_calc_class.return_value = mock_calc

        mace = MACE(symbols=["H", "H"], model_paths="fake_path", calculator=mock_calc, device="cpu")

        mace.atoms = MagicMock()
        mace.atoms.get_potential_energy.return_value = -1.0
        mace.atoms.get_forces.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        mock_calc.get_hessian.return_value = np.ones((6, 6))

        x = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        e, g, h = mace(x, task=Task.FREQ)

        ev_to_au = Energy(1.0, "eV").get("Hartree")
        ang_to_au = Length(1.0, "A").get("Bohr")

        assert e == -1.0 * ev_to_au
        expected_g = -np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]) * ev_to_au / ang_to_au
        np.testing.assert_allclose(g, expected_g)

        expected_h = np.ones((6, 6)) * ev_to_au / (ang_to_au**2)
        np.testing.assert_allclose(h, expected_h)
        assert h.shape == (6, 6)


@patch("pyrinst.potentials.mace.VibrationsData")
def test_mace_freq_modes(mock_vib_data_class):
    with patch("pyrinst.potentials.mace.MACECalculator") as mock_calc_class:
        mock_calc = MagicMock()
        mock_calc_class.return_value = mock_calc

        mace = MACE(symbols=["H", "H"], model_paths="fake_path", calculator=mock_calc, device="cpu")
        mace.atoms.calc.get_hessian = MagicMock(return_value=np.zeros((6, 6)))

        mock_vib_data = MagicMock()
        mock_vib_data_class.from_2d.return_value = mock_vib_data
        mock_vib_data.get_frequencies.return_value = np.array([100.0, 200.0])
        mock_vib_data.get_modes.return_value = np.zeros((2, 6))

        x = np.zeros((2, 3))
        freqs = mace.freq_modes(x)

        np.testing.assert_allclose(freqs, [100.0, 200.0])
        np.testing.assert_allclose(mace.freq, [100.0, 200.0])
        mock_vib_data_class.from_2d.assert_called_once()
