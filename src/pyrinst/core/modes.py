import logging
import math
import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable
from warnings import warn

import numpy as np
from numpy.linalg import norm
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from .rp import Beads, Springs
from .pes.abc import PES, PESProxy
from ..io.xyz import save
from ..utils.coordinates import mass_weight
from ..utils.mechanics import inertia
from ..utils.units import Energy, Length, Time, Temperature
from ..io.formats import Formats, format_array

log = logging.getLogger(__name__)
logging.captureWarnings(True)

hbar: float = 1


class Data(PESProxy, ABC):
    """
    todo: shape
    """
    order = None

    def __init__(self, x: NDArray, pes: PES, phase: str = 'solid'):
        assert phase in ('solid', 'liquid', 'gas')
        super().__init__(pes)
        self.x: NDArray = x
        self.phase: str = phase
        self.n_zero: int = self.get_n_zero(phase)
        self.pot, self.grad, self.hess = self.all(x)
        self.freq: NDArray | None = None  # will be evaluated after optimization
        self.normal_modes: NDArray | None = None
        self.log_pf: dict[tuple[float, int | None], tuple[float, float, float]] = {}

    def get_n_zero(self, phase: str) -> int:
        match phase:
            case 'solid':
                return 0
            case 'liquid':
                return 3
            case 'gas':
                return 3 if len(self.atoms) == 1 else (5 if self.is_linear else 6)
        raise ValueError(f'unknown phase: {phase}')

    @property
    def is_linear(self) -> bool:
        if self.x.shape[0] < 3:
            return True
        x = self.x - self.x[0]
        for i in range(2, x.shape[0]):
            if not math.isclose(norm(np.cross(x[1], x[i])), 0):
                return False
        return True

    @property
    def dof(self) -> int:
        return self.x.size

    @abstractmethod
    def update_link(self, **kwargs) -> None:
        """"""

    def recalc(self, x: NDArray, update: Callable | None = None) -> None:
        x_old, grad_old, self.x = self.x, self.grad, x
        if update:
            self.pot, self.grad = self.both(self.x)
            self.hess = update(self.hess, (self.x - x_old).ravel(), (self.grad - grad_old).ravel())
        else:
            self.pot, self.grad, self.hess = self.all(self.x)

    def move(self, dx: NDArray, update: Callable | None = None) -> None:
        self.recalc(self.x + dx, update)

    def __str__(self):
        """used for optimization only"""
        return f'V = {self.pot:{Formats.ENERGY}}, |G| = {norm(self.grad):{Formats.GRAD_NORM}}'

    def output(self, prefix: str) -> None:
        # todo: save traj
        comment = f'V = {self.pot:{Formats.ENERGY}}' if self.pot is not None else ''
        if self._pes.atoms is None:
            np.savetxt(prefix+'.txt', self.x, fmt='%15.8f', header=comment)
        else:
            save(prefix+'.xyz', self.x, self.atoms, comment)

    def final_output(self, prefix: str) -> None:
        self.recalc_hess()
        self.print_freq(raise_error=False)
        self.save(prefix)

    def save(self, prefix: str) -> None:
        self.output(prefix)
        with open(prefix+'.pkl', 'wb') as f:
            pickle.dump(self, f)

    def recalc_hess(self) -> None:
        self.hess = self.hessian(self.x)
        hess_mw = mass_weight(self.hess, self.mass, dim=self.dof//self.mass.size)
        evals, self.normal_modes = np.linalg.eigh(hess_mw)
        self.freq = np.sqrt(abs(evals)) * np.sign(evals)

    def print_freq(self, raise_error: bool) -> None:
        if len(self.freq) == self.n_zero:
            freq_nonzero = np.array([])
        else:
            freq_nonzero = self.freq[np.argpartition(np.abs(self.freq), self.n_zero)[self.n_zero:]]
        zpe = 0.5 * hbar * np.sum(freq_nonzero, where=freq_nonzero > 0)
        freq_cm: NDArray = self.freq * hbar * Energy(1, 'au').get("cm-1")
        log.info(f'frequencies in cm-1:\n{format_array(freq_cm, fmt=Formats.FREQUENCY)}')
        log.info(f'H.O. ZPE = {Energy(zpe, "au").get("cm-1"):{Formats.FREQUENCY}} cm-1')
        # check for negative eigenvalues
        if (n := sum(freq_nonzero < 0)) != self.order:
            msg: str = f'Wrong number of negative eigenvalues (expected {self.order}, got {n} instead)'
            if raise_error:
                log.error(msg)
                raise ValueError(msg)
            else:
                warn(msg, RuntimeWarning)

    @abstractmethod
    def trans(self, beta: float) -> float:
        """"""

    @abstractmethod
    def rot(self, beta: float) -> float:
        """"""

    @abstractmethod
    def vib(self, beta: float, n: int | None = None) -> float:
        """"""

    def calc_pf(self, beta: float, n: int | None = None) -> None:
        self.log_pf[(beta, n)] = (self.trans(beta), self.rot(beta), self.vib(beta, n))


class Minimum(Data):
    order = 0

    def update_link(self, **kwargs) -> None:
        if len(kwargs):
            raise ValueError('Minimum does not have link')

    def trans(self, beta: float) -> float:
        res: float = (math.sqrt(sum(self.mass)/(2*np.pi*beta))/hbar)**3 if self.n_zero >= 3 else 1
        log.info(f'Z_trans = {res:{Formats.PARTITION_FUNCTION}} per volume')
        return math.log(res)

    def rot(self, beta: float, mass: float | NDArray = None) -> float:
        mass = self.mass if mass is None else mass
        if self.n_zero >= 5:
            pmi: NDArray = np.linalg.eigvalsh(inertia(self.x, mass))  # principal moments of inertia
            pmi = np.delete(pmi, np.isclose(pmi, 0))
            rot_const: NDArray = hbar ** 2 / (2 * pmi)
            log.info(f'Moments of Inertia = {format_array(pmi, Formats.MOMENTUM_OF_INERTIA)}')
            log.info(f'Rotational Constants = {format_array(rot_const, Formats.ROTATIONAL_CONSTANT)}')
            res = 1./(pmi[1]*beta) if self.n_zero == 5 else math.sqrt(np.pi/(math.prod(pmi)*beta**3))
        else:
            res: float = 1
        log.info(f'Z_rot = {res:{Formats.PARTITION_FUNCTION}}')
        return math.log(res)

    def vib(self, beta: float, n: int | None = None) -> float:
        self.print_freq(raise_error=True)
        freq: NDArray = self.freq[self.real_freq_slice]
        if n is None:
            msg: str = '(exact harmonic)'
        else:
            freq = 2 * n / (beta * hbar) * np.arcsinh(beta * hbar * freq / (2 * n))
            msg = f'({n}-bead approx)'
        res: float = - sum(np.log(2 * np.sinh(0.5 * beta * hbar * freq)))
        log.info(f'{msg} log(Z_vib) = {res:{Formats.LOG_PARTITION_FUNCTION}}')
        return res

    @property
    def real_freq_slice(self) -> slice:
        return slice(self.n_zero, None)

    def calc_rate(self, beta: float, n: int | None = None) -> None:
        log.info(f'\n{"-"*9}\nMinimum\n{"-"*9}')
        log.info(f'min V = {self.pot:{Formats.ENERGY}}')
        self.calc_pf(beta, n)


class TransitionState(Minimum):
    order = 1

    def __init__(self, *args, rct: Minimum = None, rct2: Minimum = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta_c: float | None = None  # will be evaluated after optimization
        self.rct = self.rct2 = None
        self.update_link(rct=rct, rct2=rct2)

    def update_link(self, **kwargs) -> None:
        self.rct = kwargs.get('rct')
        self.rct2 = kwargs.get('rct2')

    def final_output(self, prefix: str) -> None:
        self.recalc_hess()
        self.print_freq(raise_error=False)
        self.beta_c = 2 * np.pi / (hbar * (-self.freq[0]))
        fmt = Formats.TEMPERATURE
        logging.info(f'such that beta_c = {self.beta_c:{fmt}}, T_c = {Temperature.to_kelvin(self.beta_c):{fmt}} K')
        self.save(prefix)

    @property
    def real_freq_slice(self) -> slice:
        return slice(self.n_zero + 1, None)

    def calc_rate(self, beta: float, n: int | None = None) -> None:
        # partition functions
        if self.rct:
            self.rct.calc_rate(beta, n)
            log_pf_rct: NDArray = np.array(self.rct.log_pf[(beta, n)])
            if self.rct2:
                self.rct2.calc_rate(beta, n)
                log_pf_rct *= np.array(self.rct2.log_pf[(beta, n)])
        else:
            log_pf_rct = np.zeros(3)
        log.info(f'\n{"-" * 18}\nTransition State\n{"-" * 18}')
        msg: str = f'TS V = {self.pot:{Formats.ENERGY}}'
        if self.rct:
            barrier: float | None = self.pot - self.rct.pot
            if self.rct2:
                barrier -= self.rct2.pot
            msg += f'; barrier = {barrier:{Formats.ENERGY}}'
        else:
            barrier = None
        log.info(msg)
        self.calc_pf(beta, n)
        if self.rct:
            log_pf: NDArray = np.array(self.log_pf[(beta, n)]) - log_pf_rct  # partition functions
            assert math.isclose(log_pf[0], 0), 'Ratio between Z_trans should be 1 for a unimolecular reaction'
            log.info('\nComputing Eyring TST rate from TS and reactant minima...')
            pf: NDArray = np.exp(log_pf)
            log.info('Partition functions (TS/reactant):')
            fmt: str = Formats.PARTITION_FUNCTION
            log.info(f'  trans {pf[0]:{fmt}}\n  rot   {pf[1]:{fmt}}\n  vib   {pf[2]:{fmt}}')
            # todo: symmetry factor
            beta_hbar: float = beta * hbar
            k_eyring: float = 1 / (2 * np.pi * beta_hbar) * math.exp(sum(log_pf) - beta * barrier)
            k_pb = k_eyring * 0.5 * beta_hbar * (-self.freq[0]) / math.sin(0.5 * beta_hbar * (-self.freq[0]))
            if self.rct2:  # bimolecular
                unit: float = Length(1, 'au').get('cm') ** 3 / Time(1, 'au').get('s')
                k_eyring_si: float = k_eyring * unit
                log.info(f'kEyring = {k_eyring:{Formats.RATE}} = {k_eyring_si:{Formats.RATE}} cm^3 / s')
                log.info(f'log10(kEyring / (cm^3 s^-1)) = {math.log10(k_eyring_si):{Formats.LOG_RATE}}')
                if beta < self.beta_c:  # exact tunneling correction for the parabolic barrier
                    k_pb_si: float = k_pb * unit
                    log.info(f'kPB = {k_pb:{Formats.RATE}} = {k_pb_si:{Formats.RATE}} cm^3 / s')
                    log.info(f'log10(kPB / (cm^3 s^-1)) = {math.log10(k_pb_si):{Formats.LOG_RATE}}')
            else:  # unimolecular
                k_eyring_si: float = k_eyring / Time(1, 'au').get('s')
                log.info(f'kEyring = {k_eyring:{Formats.RATE}} = {k_eyring_si:{Formats.RATE}} s-1')
                log.info(f'log10(kEyring / s^-1) = {math.log10(k_eyring_si):{Formats.LOG_RATE}}')
                if beta < self.beta_c:  # exact tunneling correction for the parabolic barrier
                    k_pb_si: float = k_pb / Time(1, 'au').get('s')
                    log.info(f'kPB = {k_pb:{Formats.RATE}} = {k_pb_si:{Formats.RATE}} s-1')
                    log.info(f'log10(kPB / s^-1) = {math.log10(k_pb_si):{Formats.LOG_RATE}}')

    def spread(self, n: int, beta: float, length: float = 0.1) -> 'Instanton':
        mode: NDArray = self.normal_modes[:, 0].reshape(self.x.shape) / np.sqrt(self.mass)  # un-mass-weighted mode
        mode /= norm(mode)  # renormalize
        x_inst: NDArray = self.x + length * mode[None] * np.cos(np.linspace(0, math.pi, n//2))[:, None]
        return Instanton(x_inst, self._pes, beta, self.phase, ts=self)


class Instanton(Minimum):
    """half-ring"""
    order = 1

    def __init__(
            self, x: NDArray, pes: PES, beta: float,
            phase: str = 'solid', rct: Minimum = None, rct2: Minimum = None, ts: TransitionState = None):
        self.n = 2 * len(x)
        self.beta = beta
        self.beads = Beads(pes)
        self.springs = Springs(self.n, beta, pes.mass)
        self.pot_cl: NDArray | None = None
        self.grad_cl: NDArray | None = None
        self.hess_cl: NDArray | None = None
        super().__init__(x, pes, phase)
        self.rct = self.rct2 = self.ts = None
        self.update_link(rct=rct, rct2=rct2, ts=ts)

    @property
    def dof(self) -> int:
        return self.x[0].size

    def update_link(self, **kwargs) -> None:
        if 'ts' in kwargs:
            self.ts = kwargs['ts']
            self.rct = self.ts.rct = kwargs.get('rct') or self.ts.rct
            self.rct2 = self.ts.rct2 = kwargs.get('rct2') or self.ts.rct2
        else:
            self.rct = kwargs.get('rct')
            self.rct2 = kwargs.get('rct2')

    def interpolate(self, n: int) -> None:
        indices_old, indices_new = np.linspace(0, 1, self.n//2), np.linspace(0, 1, n//2)
        self.pot_cl = CubicSpline(indices_old, self.pot_cl, extrapolate=False)(indices_new)
        self.grad_cl = CubicSpline(indices_old, self.grad_cl, extrapolate=False)(indices_new)
        self.hess_cl = CubicSpline(indices_old, self.hess_cl, extrapolate=False)(indices_new)
        self.n = n
        self.springs = Springs(n, self.beta, self.mass)

    def potential(self, x: NDArray) -> float:
        return 2 * sum(self.beads.potential(x)) + self.springs.potential(x)

    def gradient(self, x: NDArray) -> NDArray:
        return 2 * self.beads.gradient(x) + self.springs.gradient(x)

    def hessian(self, x: NDArray) -> NDArray:
        res: NDArray = self.springs.hessian(x).reshape(self.n//2, self.dof, self.n//2, self.dof)
        indices: NDArray = np.arange(len(res))
        res[indices, :, indices, :] += 2 * self.beads.hessian(x)
        return res.reshape(x.size, x.size)

    @property
    def hess_full(self) -> NDArray:
        res: NDArray = self.springs.hessian_full(self.x).reshape(self.n, self.dof, self.n, self.dof)
        indices: NDArray = np.arange(len(res))
        res[indices, :, indices, :] += np.r_[self.hess_cl, self.hess_cl[::-1]]
        return res.reshape(self.n*self.dof, self.n*self.dof)

    def both(self, x: NDArray) -> tuple[float, NDArray]:
        pot, grad = self.beads.both(x)
        return 2 * sum(pot) + self.springs.potential(x), 2 * grad + self.springs.gradient(x)

    def all(self, x: NDArray) -> tuple[float, NDArray, NDArray]:
        pot, grad, hess = self.beads.all(x)
        res: NDArray = self.springs.hessian(x).reshape(self.n // 2, self.dof, self.n // 2, self.dof)
        indices: NDArray = np.arange(len(res))
        res[indices, :, indices, :] += 2 * hess
        return 2*sum(pot)+self.springs.potential(x), 2*grad+self.springs.gradient(x), res.reshape(x.size, x.size)

    def output(self, prefix: str) -> None:
        comment = ''  # todo
        np.savetxt(prefix+'.txt', self.x, fmt='%15.8f', header=comment)

    def final_output(self, prefix: str) -> None:
        bn_atom: NDArray = np.atleast_1d(self.path_sq_disp)
        bn: float = sum(bn_atom)
        n_bn: float = self.n * bn
        bn_beta: float = n_bn / (self.beta * hbar)
        fmt: str = Formats.BN
        log.info(f'mass-weighted BN: BN = {bn:{fmt}}, N*BN = {n_bn:{fmt}}, BN/(betaN*hbar) = {bn_beta:{fmt}}')
        if self.atoms is not None:
            log.info('Contributions to BN (squared mass-weighted path length) from various atoms:')
            for a, atom in enumerate(self.atoms):
                log.info(f'atom {a} ({atom}): {np.sum(bn_atom[a])/bn:>5.1%}')
        log.info('computing bead potentials, gradients and Hessians...')
        self.recalc_hess()
        log.info(f'S/hbar = {self.action/hbar:{Formats.ACTION}}')
        self.save(prefix)

    def recalc_hess(self) -> None:
        self.pot_cl, self.grad_cl, self.hess_cl = self.beads.all(self.x)
        self.hess = self.hessian(self.x)

    @property
    def action(self) -> float:
        return self.beta / self.n * hbar * self.pot

    @property
    def energy(self) -> float:
        return (2 * sum(self.pot_cl) - self.springs.potential(self.x)) / self.n

    @property
    def path_sq_disp(self) -> float | NDArray:
        """squared displacement of the path, i.e., BN,
        given as contributions of each atom for real systems and as a single number for model systems.
        """
        return np.sum(self.mass * 2 * np.sum(np.diff(self.x, axis=0)**2, axis=0), axis=-1)

    def vib(self, beta: float, n: int | None = None) -> float:
        # BN
        bn: float = sum(np.atleast_1d(self.path_sq_disp))
        if math.isclose(self.n * bn, 0):
            raise RuntimeError('Your instanton beads are likely collapsed')
        # vibrations
        lam: NDArray = np.linalg.eigvalsh(mass_weight(self.hess_full, self.mass, dim=self.x[0].size))
        self.freq = np.sqrt(abs(lam)) * np.sign(lam)
        freq_nonzero: NDArray = self.freq[np.argpartition(abs(self.freq), self.n_zero+1)[self.n_zero+1:]]
        freq_12_cm: NDArray = self.freq[:12] * hbar * Energy(1, 'au').get("cm-1")
        log.info(f'first 12 frequencies in cm-1:\n{format_array(freq_12_cm, fmt=Formats.FREQUENCY)}')
        order: int = sum(freq_nonzero < 0)
        if order == 2 and Energy(freq_nonzero[1], 'au').get("cm-1") < 100:
            raise NotImplemented
        elif order != 1:
            raise RuntimeError(f'Wrong number of imaginary frequencies (expected 1, got {order} instead)')
        beta_n: float = beta / self.n
        res = - sum(np.log(beta_n * hbar * abs(freq_nonzero))) + (self.n_zero + 1) * math.log(self.n)
        res += 0.5 * (math.log(2 * np.pi * bn) - math.log(beta_n * hbar ** 2))
        log.info(f'log(Z_vib) = {res:{Formats.LOG_PARTITION_FUNCTION}}')
        return res

    def calc_rate(self, beta: float, n: int | None = None) -> None:
        assert math.isclose(beta, self.beta)
        n = self.n
        # partition functions
        if self.ts:
            self.ts.calc_rate(beta, n)
            log_pf_ts: NDArray = np.array(self.ts.log_pf[(beta, n)])
        else:
            if self.rct:
                self.rct.calc_rate(beta, n)
                if self.rct2:
                    self.rct2.calc_rate(beta, n)
            log_pf_ts = np.zeros(3)
        if self.rct:
            log_pf_rct: NDArray = np.array(self.rct.log_pf[(beta, n)])
            if self.rct2:
                log_pf_rct *= self.rct2.log_pf[(beta, n)]
        else:
            log_pf_rct = np.zeros(3)
        log.info(f'\n{"-"*11}\nInstanton\n{"-"*11}')
        log.info(f'V_turn = ({self.pot_cl[0]:{Formats.ENERGY}}, {self.pot_cl[-1]:{Formats.ENERGY}})')
        log.info(f'E = {self.energy:{Formats.ENERGY}}')
        self.calc_pf(beta, n)
        log_pf_inst: NDArray = np.array(self.log_pf[(beta, n)])
        fmt: str = Formats.PARTITION_FUNCTION
        if self.ts:
            log_pf: NDArray = log_pf_inst - log_pf_ts
            assert math.isclose(log_pf[0], 0), 'Ratio between Z_trans should be 1'
            log.info('\nComputing instanton tunnelling factor from instanton and TS data...')
            pf: NDArray = np.exp(log_pf)
            log.info('Partition functions (instanton/TS):')
            log.info(f'  trans {pf[0]:{fmt}}\n  rot   {pf[1]:{fmt}}\n  vib   {pf[2]:{fmt}}')
            action: float = self.action / hbar - beta * self.ts.pot
            log.info(f'S/hbar - beta*V_TS = {action:{Formats.ACTION}}')
            log.info(f'kInst/kEyring = {math.exp(sum(log_pf) - action):{Formats.TUNNELING_FACTOR}}')
        if self.rct:
            log_pf: NDArray = log_pf_inst - log_pf_rct
            if not self.rct2:
                assert math.isclose(log_pf[0], 0), 'Ratio between Z_trans should be 1 for a unimolecular reaction'
            log.info('\nComputing thermal instanton rate from instanton and reactant minima...')
            pf: NDArray = np.exp(log_pf)
            log.info('Partition functions (instanton/reactant):')
            log.info(f'  trans {pf[0]:{fmt}}\n  rot   {pf[1]:{fmt}}\n  vib   {pf[2]:{fmt}}')
            action: float = self.action / hbar - beta * self.rct.pot
            if self.rct2:
                action -= beta * self.rct2.pot
            log.info(f'S/hbar - beta*V_r = {action:{Formats.ACTION}}')
            k: float = 1 / (2 * np.pi * beta * hbar) * math.exp(sum(log_pf) - action)
            if self.rct2:
                k_si: float = k * Length(1, 'au').get('cm') ** 3 / Time(1, 'au').get('s')
                log.info(f'kInst = {k:{Formats.RATE}} = {k_si:{Formats.RATE}} cm^3 / s')
                log.info(f'log10(kInst / (cm^3 s^-1)) = {math.log10(k_si):{Formats.LOG_RATE}}')
            else:
                k_si: float = k / Time(1, 'au').get('s')
                log.info(f'kInst = {k:{Formats.RATE}} = {k_si:{Formats.RATE}} / s')
                log.info(f'log10(kInst / s^-1) = {math.log10(k_si):{Formats.LOG_RATE}}')
