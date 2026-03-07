import numpy as np
from ase import Atoms
from ase.vibrations.data import VibrationsData
from mace.calculators import MACECalculator
from numpy.typing import NDArray

from pyrinst.io.xyz import load
from pyrinst.potentials import Potential, Task
from pyrinst.utils.elements import element_data
from pyrinst.utils.units import Energy, Length, Mass


class MACE(Potential):
    """
    Wrap MACECalculator so that PES abc uses MACE energy/forces/hessian directly.

    Supports initialization from:
      - ASE Atoms object (original method)
      - XYZ file path (new method)
    """

    def __init__(self, atoms: Atoms | str, calculator: MACECalculator | None = None, **calc_kwargs):
        """
        Parameters
        ----------
        atoms : Atoms | str
            Either an ASE Atoms object OR a path to an XYZ file.
        calculator : MACECalculator | None
            Pre-initialized MACE calculator. If None, creates one using calc_kwargs.
        **calc_kwargs
            Arguments for MACECalculator if calculator is None.
        """
        if isinstance(atoms, str):
            coords, symbols = load(atoms, return_symbols=True)

            if coords.ndim == 3:
                coords = coords[0]

            coords *= Length(1, "au").get("A")
            atoms = Atoms(symbols=symbols.tolist(), positions=coords, calculator=calculator)
        else:
            atoms = atoms

        if calculator is None:
            self.calculator = MACECalculator(**calc_kwargs)
        else:
            self.calculator = calculator

        self._atoms_template = atoms.copy()
        self.atoms = list(atoms.get_chemical_symbols())
        self.mass = element_data.get_masses(self.atoms) * Mass(1, "amu").get("au")

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
