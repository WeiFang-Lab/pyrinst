from collections.abc import Sequence

import numpy as np
from ase import Atoms
from ase.vibrations.data import VibrationsData
from mace.calculators import MACECalculator
from numpy.typing import NDArray

from pyrinst.potentials import Potential, Task
from pyrinst.utils.units import Energy, Length


class MACE(Potential):
    """
    Wrap MACECalculator so that PES abc uses MACE energy/forces/hessian directly.

    Supports initialization from:
      - ASE Atoms object (original method)
      - XYZ file path (new method)
    """

    def __init__(
        self,
        symbols: Sequence[str],
        model_paths: str,
        default_dtype="float64",
        device="cuda",
        enable_cueq=False,
        calculator: MACECalculator | None = None,
        **_,
    ):
        """
        Parameters
        ----------
        symbols : Sequence[str]
            List of element symbols.
        calculator : MACECalculator | None
            Pre-initialized MACE calculator. If None, creates one using calc_kwargs.
        """
        symbols = symbols.tolist() if isinstance(symbols, np.ndarray) else list(symbols)
        self.atoms = Atoms(symbols=symbols, positions=np.empty((len(symbols), 3)))

        if calculator is None:
            calculator = MACECalculator(
                model_paths=model_paths, default_dtype=default_dtype, device=device, enable_cueq=enable_cueq
            )

        self.atoms.calc = calculator
        self.symbols = symbols

    def __call__(self, x: NDArray, task: Task = Task.SP) -> tuple[float, None, NDArray | None]:
        arr = np.asarray(x, dtype=float).reshape(-1, 3) * Length(1, "au").get("A")
        self.atoms.set_positions(arr)
        energy_conv = Energy(1.0, "eV").get("Hartree")
        length_conv = Length(1.0, "A").get("Bohr")
        energy = self.atoms.get_potential_energy() * energy_conv
        gradient = -self.atoms.get_forces() * energy_conv / length_conv if task > Task.SP else None
        if task > Task.GRAD:
            hessian = self.atoms.calc.get_hessian(atoms=self.atoms).reshape(x.size, x.size)
            hessian *= energy_conv / (length_conv**2)
            hessian = 0.5 * (hessian + hessian.T)
        else:
            hessian = None
        return energy, gradient, hessian

    def freq_modes(self, x: NDArray) -> NDArray:
        arr = np.asarray(x, dtype=float).reshape(-1, 3) * Length(1, "au").get("A")
        self.atoms.set_positions(arr)

        hess = self.calculator.get_hessian(atoms=self.atoms)
        hess_2d = hess.reshape(3 * len(self.atoms), 3 * len(self.atoms))

        indices = np.arange(len(self.atoms))
        vib_data = VibrationsData.from_2d(self.atoms, hess_2d, indices=indices)

        self.normal_modes = vib_data.get_modes(all_atoms=True)
        self.freq = vib_data.get_frequencies()
        print(self.normal_modes, self.freq)
        return vib_data.get_frequencies()
