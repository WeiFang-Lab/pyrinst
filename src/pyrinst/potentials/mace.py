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

    def __init__(self, symbols: Sequence[str], calculator: MACECalculator | None = None, **calc_kwargs):
        """
        Parameters
        ----------
        symbols : Sequence[str]
            List of element symbols.
        calculator : MACECalculator | None
            Pre-initialized MACE calculator. If None, creates one using calc_kwargs.
        **calc_kwargs
            Arguments for MACECalculator if calculator is None.
        """
        symbols = symbols.tolist() if isinstance(symbols, NDArray) else list(symbols)
        atoms = Atoms(symbols=symbols, positions=np.empty((len(symbols), 3)))

        if calculator is None:
            calc_kwargs["model_paths"] = calc_kwargs["template_input"]
            self.calculator = MACECalculator(**calc_kwargs)
        else:
            self.calculator = calculator

        self._atoms_template = atoms.copy()

    def __call__(self, x: NDArray, task: Task = Task.SP) -> tuple[float, None, NDArray | None]:
        return self.potential(x), None, self.hessian(x)

    def _atoms_from_x(self, x: NDArray) -> Atoms:
        arr = np.asarray(x, dtype=float).reshape(-1, 3) * Length(1, "au").get("A")
        atoms = self._atoms_template.copy()
        atoms.set_positions(arr)
        atoms.calc = self.calculator
        return atoms

    def potential(self, x: NDArray) -> float:
        atoms = self._atoms_from_x(x)
        return Energy(atoms.get_potential_energy(), "eV").get("Hartree")

    def hessian(self, x: NDArray) -> NDArray:

        atoms = self._atoms_from_x(x)
        hess = self.calculator.get_hessian(atoms=atoms)

        dof = x.size
        hess = hess.reshape(dof, dof)

        energy_conv = Energy(1.0, "eV").get("Hartree")
        length_conv = Length(1.0, "A").get("Bohr")
        factor = energy_conv / (length_conv**2)
        hess_au = hess * factor

        hess_au = 0.5 * (hess_au + hess_au.T)
        return hess_au

    def freq_modes(self, x: NDArray) -> NDArray:

        atoms = self._atoms_from_x(x)

        hess = self.calculator.get_hessian(atoms=atoms)
        hess_2d = hess.reshape(3 * len(atoms), 3 * len(atoms))

        indices = np.arange(len(atoms))
        vib_data = VibrationsData.from_2d(atoms, hess_2d, indices=indices)

        self.normal_modes = vib_data.get_modes(all_atoms=True)
        self.freq = vib_data.get_frequencies()
        print(self.normal_modes, self.freq)
        return vib_data.get_frequencies()
