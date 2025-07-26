import logging
import math
import pickle
from abc import ABC
from warnings import warn

import numpy as np
from numpy.linalg import norm
from numpy.typing import NDArray

from pyrinst.config.formats import FORMATS
from pyrinst.config.units import UnitSystem
from pyrinst.core.opt.hessian import bofill
from pyrinst.core.pes.abc import PES
from pyrinst.utils.coordinates import mass_weight

log = logging.getLogger(__name__)
logging.captureWarnings(True)


class Data(ABC):
    """
    todo: shape
    """
    order = None

    def __init__(self, x: NDArray, pes: PES, n_zero: int = 0):
        assert n_zero in (0, 3, 5, 6)
        self.x = x
        self.pes = pes
        self.mass: float | NDArray = pes.mass
        self.units: UnitSystem = pes.units
        self.hbar: float = self.units.hbar
        self.n_zero = n_zero
        self.pot, self.grad, self.hess = pes.all(x)
        self.freq: NDArray | None = None  # will be evaluated after optimization

    def update(self, x: NDArray, calc_hess: bool = False) -> None:
        x_old, grad_old, self.x = self.x, self.grad, x
        if calc_hess:
            self.pot, self.grad, self.hess = self.pes.all(self.x)
        else:
            self.pot, self.grad = self.pes.both(self.x)
            self.hess = bofill(self.hess, (self.x - x_old).ravel(), (self.grad - grad_old).ravel())

    def move(self, dx: NDArray) -> None:
        self.update(self.x + dx)

    def __str__(self):
        """used for optimization only"""
        return f'V = {self.pot:{FORMATS["energy"]}}, |G| = {norm(self.grad):.5e}'

    def output(self, prefix: str) -> None:
        # todo: save traj, xyz
        comment = f'V = {self.pot:{FORMATS["energy"]}}' if self.pot is not None else ''
        np.savetxt(prefix+'.txt', self.x, fmt='%15.8f', header=comment)

    def final_output(self, prefix: str) -> None:
        self.hess = self.pes.hessian(self.x)
        self.output(prefix)
        hess_mw = mass_weight(self.hess, self.mass, dim=self.x.size//self.mass.size)
        evals = np.linalg.eigvalsh(hess_mw)  # todo: project
        self.freq = np.sqrt(abs(evals)) * np.sign(evals)
        self.print_freq(raise_error=False)
        with open(prefix+'.pkl', 'wb') as f:
            pickle.dump(self, f)

    def print_freq(self, raise_error: bool) -> None:
        freq_nonzero = self.freq[np.argpartition(np.abs(self.freq), self.n_zero)[self.n_zero:]]
        zpe = 0.5 * self.hbar * np.sum(freq_nonzero, where=freq_nonzero > 0)
        log.info(f'frequencies in cm-1:\n{self.units.Energy(self.freq).get("cm-1")}')  # todo: fmt
        log.info(f'H.O. ZPE = {self.units.Energy(zpe).get("cm-1"):.4f} cm-1')
        # check for negative eigenvalues
        if (n := sum(freq_nonzero < 0)) != self.order:
            msg: str = f'Wrong number of negative eigenvalues (expected {self.order}, got {n} instead)'
            if raise_error:
                log.error(msg)
                raise ValueError(msg)
            else:
                warn(msg, RuntimeWarning)


class Minimum(Data):
    order = 0

    def trans(self, beta: float) -> float:
        res: float = 1
        log.info(f'Z_trans = {res:{FORMATS['pf']}} per volume')
        return math.log(res)  # todo

    def rot(self, beta: float) -> float:
        res: float = 1
        log.info(f'Z_rot = {res:{FORMATS["pf"]}}')
        return math.log(res)  # todo

    def vib(self, beta: float) -> float:
        self.print_freq(raise_error=True)
        # todo: finite N
        res: float = - sum(np.log(2 * np.sinh(0.5 * beta * self.hbar * self.freq[self.n_zero:])))
        log.info(f'log(Z_vib) = {res:{FORMATS["log pf"]}}')
        return res

    def fluctuation(self, beta: float) -> tuple[float, float, float]:
        return self.trans(beta), self.rot(beta), self.vib(beta)

    def calc_rate(self, beta: float) -> None:
        log.info(f'\n{"-"*9}\nMinimum\n{"-"*9}')
        log.info(f'min V = {self.pot:{FORMATS["energy"]}}')
        self.fluctuation(beta)


class TransitionState(Minimum):
    order = 1

    def __init__(self, x: NDArray, pes: PES, n_zero: int = 0, reactant: Minimum = None):
        super().__init__(x, pes, n_zero)
        self.beta_c: float | None = None  # will be evaluated after optimization
        self.reactant = reactant  # todo: bimolecular

    def final_output(self, prefix: str) -> None:
        super().final_output(prefix)
        self.beta_c = 2 * np.pi / (self.hbar * (-self.freq[0]))
        fmt = FORMATS['temperature']
        logging.info(f'such that beta_c = {self.beta_c:{fmt}}, T_c = {self.units.betaTemp(self.beta_c):{fmt}} K')

    def vib(self, beta: float) -> float:
        self.print_freq(raise_error=True)
        # todo: finite N
        res: float = - sum(np.log(2 * np.sinh(0.5 * beta * self.hbar * self.freq[self.n_zero+1:])))
        log.info(f'log(Z_vib) = {res:{FORMATS["log pf"]}}')
        return res

    def calc_rate(self, beta: float) -> None:
        # partition functions
        log_pf_r: float | NDArray = np.array(self.reactant.fluctuation(beta)) if self.reactant else 0
        log.info(f'\n{"-" * 9}\nTransition State\n{"-" * 9}')
        msg: str = f'TS V = {self.pot:{FORMATS["energy"]}}'
        if self.reactant:
            barrier: float | None = self.pot - self.reactant.pot
            msg += f'; barrier = {barrier:{FORMATS["energy"]}}'
        else:
            barrier = None
        log.info(msg)
        log_pf: NDArray = np.array(self.fluctuation(beta)) - log_pf_r  # partition functions
        if self.reactant:
            assert math.isclose(log_pf[0], 0), 'Ratio between Z_trans should be 1 for a unimolecular reaction'
            log.info('\nComputing Eyring TST rate from TS and reactant minima...')
            pf: NDArray = np.exp(log_pf)
            log.info('Partition functions (TS/reactant):')
            fmt: str = FORMATS['pf']
            log.info(f'  trans {pf[0]:{fmt}}\n  rot   {pf[1]:{fmt}}\n  vib   {pf[2]:{fmt}}')
            # todo: symmetry factor
            beta_hbar: float = beta * self.hbar
            k_eyring: float = 1 / (2 * np.pi * beta_hbar) * math.exp(sum(log_pf) - beta * barrier)
            k_eyring_si: float = k_eyring / self.units.time
            log.info(f'kEyring = {k_eyring:{FORMATS["rate"]}} = {k_eyring_si:{FORMATS["rate"]}} s-1')
            log.info(f'log10(kEyring / s^-1) = {math.log10(k_eyring_si):{FORMATS["log rate"]}}')
            if beta < self.beta_c:  # exact tunneling correction for the parabolic barrier
                k_pb = k_eyring * 0.5 * beta_hbar * (-self.freq[0]) / math.sin(0.5 * beta_hbar * (-self.freq[0]))
                k_pb_si: float = k_pb / self.units.time
                log.info(f'kPB = {k_pb:{FORMATS["rate"]}} = {k_pb_si:{FORMATS["rate"]}} s-1')
                log.info(f'log10(kPB / s^-1) = {math.log10(k_pb_si):{FORMATS["log rate"]}}')
