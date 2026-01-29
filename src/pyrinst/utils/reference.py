import numpy as np
from numpy.typing import NDArray
from scipy import constants as sc

from pyrinst.core.pes.abc import PES
from pyrinst.core.modes import Data
from pyrinst.utils.units import Energy, Mass
from pyrinst.config.constants import KELVIN 

# parser = argparse.ArgumentParser()
# parser.add_argument('input', help='Centroid structure in xyz, txt, or pkl format.')
# parser.add_argument('-P', '--PES', choices=('MACE',), help='Potential energy surface')
# # parameters for mace
# parser.add_argument('--model_path', help='path of MACE model path', type=str)
# parser.add_argument('--dtype', help='dtype of MACE model', type=str, default='float64')

class FEPRef(Data):

    def __init__(self, x: NDArray, pes: PES, phase: str = 'gas'):
        super().__init__(x, pes, phase)
        self.ref: float
        self.masses = self.mass
        self.atoms = self.atoms

    def __getstate__(self):

        """
        Exclude pes to avoid saving heavy object
        """
        state = self.__dict__.copy()
        state.pop("_pes", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._pes = None
    
    def update_link(self, **kwargs) -> None:
        pass

    def trans(self, beta: float) -> float:
        pass

    def rot(self, beta: float) -> float:
        pass

    def vib(self, beta: float, n: int | None = None) -> float:
        pass
    
    def norm_dimensionless_modes(self) -> NDArray:
        
        modes_raw = self.normal_modes.T.reshape(self.dof, len(self.mass), 3) # shape (3N, N, 3)
        mass_amu = self.mass * Mass(1, 'au').get('amu')
        mass_factor = mass_amu[np.newaxis, :, np.newaxis] ** -0.5 # shape (1, N, 1)
        ase_modes = modes_raw * mass_factor
        self.normal_modes = ase_modes
        return self

    def max_img_freq(self):
        pass
    # def compute_ref(self, temperature: float = 300, nbeads: int = 24) -> float:
    #     """
    #     Compute the special free energy from vibrational frequencies.

    #     Parameters
    #     ----------
    #     temperature : float
    #         Temperature in K. Default 300.
    #     nbeads : int
    #         Number of beads. Default 24.

    #     Returns
    #     -------
    #     float
    #         The computed harmonic reference energy (in eV)
    #     """
    #     beta = 1.0 / (float(KELVIN) * float(temperature))

    #     if getattr(self, "freq", None) is None:
    #         raise RuntimeError("self.freq is not set. Call recalc_hess() first.")
        
    #     freqs = np.asarray(self.freq).astype(float)

    #     freqs_real = freqs[freqs > 0]
    #     # ws_hartree = np.array([Energy(float(f), 'cm-1').get('Hartree') for f in freqs_real])
    #     ws_hartree = freqs_real

    #     def ana(x: np.ndarray) -> np.ndarray:
    #         em2x = np.exp(-2.0 * x)
    #         return x + np.log(1 - em2x) - np.log(2.0 * x)

    #     def dF0(ws: np.ndarray, beta: float, N: int = 24) -> float:
    #         hbfs = (beta * ws) / 2.0
    #         arg = hbfs / float(N)
    #         temp = np.arcsinh(arg) * float(N)
    #         return float(np.sum(ana(temp)) / beta)
        
    #     print(ws_hartree)
    #     dF_hartree = dF0(ws_hartree, float(beta), N=int(nbeads))
    #     dF_eV = Energy(float(dF_hartree), 'Hartree').get('eV')
    #     print(dF_eV)
    #     self.ref = dF_eV
    #     return self