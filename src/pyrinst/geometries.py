import logging
import math
import pickle
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar
from warnings import warn

import numpy as np
from numpy.linalg import norm
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from pyrinst.io.formats import Formats, format_array
from pyrinst.io.xyz import save
from pyrinst.thermo import ThermoData
from pyrinst.utils.coordinates import mass_weight
from pyrinst.utils.elements import element_data
from pyrinst.utils.mechanics import inertia
from pyrinst.utils.units import Energy, Mass, Temperature

log = logging.getLogger(__name__)
logging.captureWarnings(True)

hbar: float = 1
GEOMETRY_REGISTRY: dict[str, type["Geometry"]] = {}


@dataclass(slots=True)
class Geometry:
    coords: NDArray
    symbols: Sequence[str] | None = None
    energy: float | None = field(default=None, init=False)
    grad: NDArray | None = field(default=None, init=False)
    hess: NDArray | None = field(default=None, init=False)

    type_alias: ClassVar[str | None] = None

    def __init_subclass__(cls):
        if cls.type_alias is not None:
            GEOMETRY_REGISTRY[cls.type_alias] = cls

    @property
    def x(self) -> NDArray:
        return self.coords

    @x.setter
    def x(self, value: NDArray) -> None:
        self.coords = value

    @property
    def V(self) -> float:
        return self.energy

    @V.setter
    def V(self, value: float) -> None:
        self.energy = value

    @property
    def G(self) -> NDArray:
        return self.grad

    @G.setter
    def G(self, value: NDArray) -> None:
        self.grad = value

    @property
    def H(self) -> NDArray:
        return self.hess

    @H.setter
    def H(self, value: NDArray) -> None:
        self.hess = value


class PhaseType(StrEnum):
    MODEL = "model"
    SOLID = "solid"
    LIQUID = "liquid"
    GAS = "gas"


@dataclass(slots=True)
class StationaryPoint(Geometry, ABC):
    n_zero: int = field(default=0)
    links: list["StationaryPoint"] = field(default_factory=list)
    masses: NDArray | None = field(default=None)

    freqs: NDArray | None = field(default=None, init=False)
    modes: NDArray | None = field(default=None, init=False)

    order: ClassVar[int | None] = None
    type_alias: ClassVar[str | None] = None

    def __post_init__(self):
        self.update_links(*self.links)
        if self.masses is None:
            self.masses = element_data.get_masses(self.symbols) * Mass(1, "amu").get("au")

    @abstractmethod
    def __hash__(self) -> int: ...

    def update_links(self, *args) -> None:
        for arg in args:
            if not isinstance(arg, StationaryPoint):
                raise ValueError(f"Invalid link type: {type(arg)}")
        args = tuple(sorted(args, key=lambda a: (hash(a), a.V)))
        if len(args) > 1 and type(args[1]) is Minimum:
            args[0].second_mol = True
        tmp: set[StationaryPoint] = {self, *args}
        for arg in args:
            tmp.update(arg.links)
        self.links = sorted(tmp, key=hash)

    @property
    def m(self) -> NDArray:
        return self.masses

    @property
    def dof(self) -> int:
        return self.x.size

    def __str__(self):
        """used for optimization only"""
        return f"V = {self.V:{Formats.ENERGY}}, |G| = {norm(self.G):{Formats.GRAD_NORM}}"

    def output(self, filename: str) -> None:
        # todo: save traj
        comment = f"V = {self.V:{Formats.ENERGY}}" if self.V is not None else ""
        if self.symbols is None:
            np.savetxt(filename + ".txt", np.squeeze(self.x), fmt="%15.8f", header=comment)
        else:
            save(filename + ".xyz", self.x, self.symbols, comment)

    def final_output(self, filename: str) -> None:
        self.calc_freq()
        self.print_freq()
        self.save(filename)

    def calc_freq(self) -> None:
        hess_mw = mass_weight(self.hess, self.m, dim=self.x.shape[-1])
        eigs, self.modes = np.linalg.eigh(hess_mw)
        self.freqs = np.sqrt(abs(eigs)) * np.sign(eigs)

    def print_freq(self) -> None:
        if len(self.freqs) == self.n_zero:
            freqs_nonzero = np.array([])
        else:
            freqs_nonzero = self.freqs[np.argpartition(np.abs(self.freqs), self.n_zero)[self.n_zero :]]
        zpe = 0.5 * hbar * np.sum(freqs_nonzero, where=freqs_nonzero > 0)
        freqs_cm: NDArray = self.freqs[:12] * hbar * Energy(1, "au").get("cm-1")
        log.info(f"frequencies in cm-1:\n{format_array(freqs_cm, fmt=Formats.FREQUENCY)}")
        log.info(f"H.O. ZPE = {Energy(zpe, 'au').get('cm-1'):{Formats.FREQUENCY}} cm-1")
        # check for negative eigenvalues
        if (n := sum(freqs_nonzero < 0)) != self.order:
            msg: str = f"Wrong number of negative eigenvalues (expected {self.order}, got {n} instead)"
            warn(msg, RuntimeWarning, stacklevel=2)

    def save(self, filename: str) -> None:
        self.output(filename)
        with open(filename + ".pkl", "wb") as f:
            pickle.dump(self, f)

    def get_thermo_data(self, beta: float, N: int | None = None) -> ThermoData:
        data = ThermoData(beta, self.type_alias)
        data.energy = self.V
        self.trans(data, beta)
        self.rot(data, beta)
        self.vib(data, beta, N)
        return data

    def trans(self, data: ThermoData, beta: float) -> None:
        data.log_pf[0] = 3 * math.log(math.sqrt(sum(self.m) / (2 * np.pi * beta)) / hbar) if self.n_zero >= 3 else 0

    def rot(self, data: ThermoData, beta: float, masses: float | NDArray = None) -> None:
        masses = self.masses if masses is None else masses
        if self.n_zero >= 5:
            pmi: NDArray = np.linalg.eigvalsh(inertia(self.x, masses))  # principal moments of inertia
            pmi = np.delete(pmi, np.isclose(pmi, 0))
            rot_const: NDArray = hbar**2 / (2 * pmi)
            data.inertia = pmi
            data.rot_const = rot_const
            if self.n_zero == 5:
                data.log_pf[1] = -math.log(rot_const[1] * beta)
            else:
                data.log_pf[1] = 0.5 * math.log(np.pi / (math.prod(rot_const) * beta**3))

    def vib(self, data: ThermoData, beta: float, N: int | None = None) -> None:
        freqs: NDArray = self.freqs[self.n_zero + self.order :]
        if N is not None:
            freqs = 2 * N / (beta * hbar) * np.arcsinh(beta * hbar * freqs / (2 * N))
        if self.freqs.size > self.n_zero:
            data.freqs = np.sort(self.freqs[np.argpartition(abs(self.freqs), self.n_zero)[self.n_zero :]])
        else:
            data.freqs = np.array([])
        data.N = N
        data.log_pf[2] = -sum(np.log(2 * np.sinh(0.5 * beta * hbar * freqs)))


@dataclass(slots=True)
class Minimum(StationaryPoint):
    second_mol: bool = False

    order: ClassVar[int] = 0
    type_alias: ClassVar[str] = "min"

    def __hash__(self) -> int:
        return -1 if self.second_mol else 0

    def update_links(self, *args) -> None:
        if len(args):
            raise ValueError("Minimum does not have link")


@dataclass(slots=True)
class TransitionState(StationaryPoint):
    order: ClassVar[int] = 1
    type_alias: ClassVar[str] = "ts"

    def __hash__(self) -> int:
        return 1

    def final_output(self, filename: str) -> None:
        StationaryPoint.final_output(self, filename)  # must explicitly call parent method if @dataclass(slots=True)
        beta_c = 2 * np.pi / (hbar * (-self.freqs[0]))
        fmt = Formats.TEMPERATURE
        log.info(f"such that beta_c = {beta_c:{fmt}}, T_c = {Temperature.to_kelvin(beta_c):{fmt}} K")

    def spread(self, N: int, beta: float, length: float = 0.1) -> "Instanton":
        # un-mass-weighted mode
        if self.modes is None:
            self.calc_freq()
        mode: NDArray = self.modes[:, 0].reshape(self.x.shape) / np.sqrt(self.m)[:, None]
        mode /= norm(mode)  # renormalize
        phase: NDArray = np.linspace(0, math.pi, N // 2)
        x_inst: NDArray = self.x + length * mode[None, ...] * np.cos(phase).reshape(-1, *(1,) * self.x.ndim)
        return Instanton(x_inst, self.symbols, n_zero=self.n_zero, links=[self], masses=self.m, beta=beta)


@dataclass(slots=True)
class Springs:
    """half-ring"""

    N: int
    beta: float
    masses: NDArray
    omega_n: float = field(init=False)

    def __post_init__(self):
        self.omega_n: float = self.N / (self.beta * hbar)

    def potential(self, x: NDArray) -> float:
        dx: NDArray = np.diff(x, axis=0)
        return self.omega_n**2 * np.einsum("j,ijk,ijk", self.masses, dx, dx)

    def gradient(self, x: NDArray) -> NDArray:
        res: NDArray = np.zeros_like(x)
        dx: NDArray = np.diff(x, axis=0)
        res[:-1] -= dx
        res[1:] += dx
        return 2 * self.omega_n**2 * self.masses[:, None] * res

    def hessian(self, x: NDArray) -> NDArray:  # todo: banded
        tmp: NDArray = (2 * np.ones_like(x[0]) * self.masses[:, None] * self.omega_n**2).ravel()
        d: int = tmp.size
        res: NDArray = np.zeros((self.N // 2 * d, self.N // 2 * d))
        indices: NDArray = np.arange(len(res))
        res[indices[:-d], indices[:-d]] = tmp[indices[:-d] % d]
        res[indices[d:], indices[d:]] += tmp[indices[d:] % d]
        res[indices[:-d], indices[d:]] = res[indices[d:], indices[:-d]] = -tmp[indices[d:] % d]
        return res

    def hessian_full(self, x: NDArray) -> NDArray:
        tmp: NDArray = (np.ones_like(x[0]) * self.masses[:, None] * self.omega_n**2).ravel()
        d: int = tmp.size
        res: NDArray = np.zeros((self.N * d, self.N * d))
        indices: NDArray = np.arange(len(res))
        res[indices, indices] = 2 * tmp[indices % d]
        res[indices, indices - d] = res[indices - d, indices] = -tmp[indices % d]
        return res


@dataclass(slots=True)
class Instanton(TransitionState):
    """half-ring instanton"""

    beta: float | None = None
    N: int = field(init=False)
    springs: Springs = field(init=False)
    type_alias: ClassVar[str] = "inst"

    def __post_init__(self):
        StationaryPoint.__post_init__(self)
        if self.beta is None:
            raise ValueError("beta must be specified for instanton")
        self.N: int = 2 * len(self.x)
        self.springs = Springs(self.N, self.beta, self.masses)

    def __hash__(self) -> int:
        return 2

    @property
    def V(self) -> float:
        return 2 * sum(self.energy) + self.springs.potential(self.x)

    @V.setter
    def V(self, value: NDArray) -> None:
        self.energy = value

    @property
    def G(self) -> NDArray:
        return 2 * self.grad + self.springs.gradient(self.x)

    @G.setter
    def G(self, value: NDArray) -> None:
        self.grad = value

    @property
    def H(self) -> NDArray:
        if self.hess.ndim == 3:
            res: NDArray = self.springs.hessian(self.x).reshape(self.N // 2, self.dof, self.N // 2, self.dof)
            indices: NDArray = np.arange(len(res))
            res[indices, :, indices, :] += 2 * self.hess
            return res.reshape(self.x.size, self.x.size)
        else:  # ndim == 2
            return self.hess

    @H.setter
    def H(self, value: NDArray) -> None:
        self.hess = value

    @property
    def hessian_full(self) -> NDArray:
        res: NDArray = self.springs.hessian_full(self.x).reshape(self.N, self.dof, self.N, self.dof)
        indices: NDArray = np.arange(len(res))
        res[indices, :, indices, :] += np.r_[self.hess, self.hess[::-1]]
        return res.reshape(self.N * self.dof, self.N * self.dof)

    @property
    def dof(self) -> int:
        return self.x[0].size

    def interpolate(self, N: int) -> None:
        indices_old, indices_new = np.linspace(0, 1, self.N // 2), np.linspace(0, 1, N // 2)
        self.x = CubicSpline(indices_old, self.x, extrapolate=False)(indices_new)
        self.energy = CubicSpline(indices_old, self.energy, extrapolate=False)(indices_new)
        self.grad = CubicSpline(indices_old, self.grad, extrapolate=False)(indices_new)
        self.hess = CubicSpline(indices_old, self.hess, extrapolate=False)(indices_new)
        self.N = N
        self.springs = Springs(self.N, self.beta, self.masses)

    def set_beta(self, beta: float) -> None:
        self.beta = beta
        self.springs = Springs(self.N, self.beta, self.masses)

    def final_output(self, prefix: str) -> None:
        contrib: NDArray = self.m * 2 * np.sum(np.sum(np.diff(self.x, axis=0) ** 2, axis=0), axis=-1)
        BN: float = np.sum(contrib)
        fmt: str = Formats.BN
        log.info(f"mass-weighted BN: BN = {BN:{fmt}}, BN/(betaN*hbar) = {self.N * BN / (self.beta * hbar):{fmt}}")
        if self.symbols is not None:
            log.info("Contributions to BN (squared mass-weighted path length) from various atoms:")
            for a, atom in enumerate(self.symbols):
                log.info(f"atom {a} ({atom}): {contrib[a] / BN:>5.1%}")
        log.info(f"S/hbar = {self.S / hbar:{Formats.ACTION}}")
        self.save(prefix)

    @property
    def S(self) -> float:
        return self.beta / self.N * hbar * self.V

    @property
    def E(self) -> float:
        """Tunneling energy"""
        return (2 * sum(self.energy) - self.springs.potential(self.x)) / self.N

    @property
    def BN(self) -> float:
        dx: NDArray = np.diff(self.x, axis=0)
        return 2 * np.einsum("j,ijk,ijk", self.m, dx, dx)

    def get_thermo_data(self, beta: float, N: int | None = None) -> ThermoData:
        data = ThermoData(beta, self.type_alias)
        data.energy = self.V / self.N
        self.trans(data, beta)
        self.rot(data, beta, 2 * self.m / self.N)
        self.vib(data, beta, N)
        return data

    def vib(self, data: ThermoData, beta: float, N: int | None = None) -> None:
        BN: float = self.BN
        if np.isclose(self.N * BN, 0):
            raise RuntimeError("Your instanton beads are likely collapsed")
        # vibrations
        lam: NDArray = np.linalg.eigvalsh(mass_weight(self.hessian_full, self.m, dim=self.x.shape[-1]))
        self.freqs: NDArray = np.sqrt(abs(lam)) * np.sign(lam)
        freqs_nonzero: NDArray = self.freqs[np.argpartition(abs(self.freqs), self.n_zero + 1)[self.n_zero + 1 :]]
        order: int = sum(freqs_nonzero < 0)
        if order == 2 and Energy(freqs_nonzero[1], "au").get("cm-1") < 100:
            raise NotImplementedError
        elif order != 1:
            raise RuntimeError(f"Wrong number of imaginary frequencies (expected 1, got {order} instead)")
        beta_n: float = beta / self.N
        res = -sum(np.log(beta_n * hbar * abs(freqs_nonzero))) + (self.n_zero + 1) * math.log(self.N)
        res += 0.5 * (math.log(2 * np.pi * BN) - math.log(beta_n * hbar**2))
        data.freqs = np.sort(freqs_nonzero)
        data.log_pf[2] = res
