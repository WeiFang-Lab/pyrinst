import logging
import math
import pickle
from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
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
from pyrinst.opt import centroid
from pyrinst.thermo import ThermoData
from pyrinst.utils.coordinates import mass_weight
from pyrinst.utils.elements import element_data
from pyrinst.utils.mechanics import inertia
from pyrinst.utils.units import HBAR, KB, Energy, Mass, Temperature

log = logging.getLogger(__name__)
logging.captureWarnings(True)

GEOMETRY_REGISTRY: dict[str, type["Geometry"]] = {}


@dataclass(slots=True)
class Geometry:
    coords: NDArray
    symbols: Sequence[str] | None = None
    n_zero: int = 0
    energy: float | None = field(default=None, init=False)
    grad: NDArray | None = field(default=None, init=False)
    hess: NDArray | None = field(default=None, init=False)
    masses: NDArray | None = field(default=None)

    freqs: NDArray | None = field(default=None, init=False)
    modes: NDArray | None = field(default=None, init=False)

    type_alias: ClassVar[str | None] = None

    def __init_subclass__(cls):
        if cls.type_alias is not None:
            GEOMETRY_REGISTRY[cls.type_alias] = cls

    def __post_init__(self):
        if self.masses is None:
            self.masses = element_data.get_masses(self.symbols) * Mass(1, "amu").get("au")

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

    @property
    def m(self) -> NDArray:
        return self.masses

    @property
    def dof(self) -> int:
        return self.x.size

    def calc_freq(self) -> None:
        hess_mw = mass_weight(self.H, self.m, dim=self.x.shape[-1])
        eigs, self.modes = np.linalg.eigh(hess_mw)
        self.freqs = np.sqrt(abs(eigs)) * np.sign(eigs)

    def save(self, filename: str) -> None:
        with open(filename + ".pkl", "wb") as f:
            pickle.dump(self, f)


class PhaseType(StrEnum):
    MODEL = "model"
    SOLID = "solid"
    LIQUID = "liquid"
    GAS = "gas"


@dataclass(slots=True)
class StationaryPoint(Geometry, ABC):
    links: list["StationaryPoint"] = field(default_factory=list)

    order: ClassVar[int | None] = None
    type_alias: ClassVar[str | None] = None

    def __post_init__(self):
        Geometry.__post_init__(self)
        self.update_links(*self.links)

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

    def __str__(self):
        """used for optimization only"""
        return f"V = {self.V:{Formats.ENERGY}}, |G| = {norm(self.G):{Formats.GRAD_NORM}}"

    def output(self, filename: str) -> None:
        comment = f"V = {self.V:{Formats.ENERGY}}" if self.V is not None else ""
        if self.symbols is None:
            np.savetxt(filename + ".txt", np.squeeze(self.x), fmt="%15.8f", header=comment)
        else:
            save(filename + ".xyz", self.x, self.symbols, comment)

    def final_output(self, filename: str) -> None:
        self.calc_freq()
        self.print_freq()
        self.save(filename)

    def print_freq(self) -> None:
        if len(self.freqs) == self.n_zero:
            freqs_nonzero = np.array([])
        else:
            freqs_nonzero = self.freqs[np.argpartition(np.abs(self.freqs), self.n_zero)[self.n_zero :]]
        zpe = 0.5 * HBAR * np.sum(freqs_nonzero, where=freqs_nonzero > 0)
        freqs_cm: NDArray = self.freqs[:12] * HBAR * Energy(1, "au").get("cm-1")
        log.info(f"frequencies in cm-1:\n{format_array(freqs_cm, fmt=Formats.FREQUENCY)}")
        log.info(f"H.O. ZPE = {Energy(zpe, 'au').get('cm-1'):{Formats.FREQUENCY}} cm-1")
        # check for negative eigenvalues
        if (n := sum(freqs_nonzero < 0)) != self.order:
            msg: str = f"Wrong number of negative eigenvalues (expected {self.order}, got {n} instead)"
            warn(msg, RuntimeWarning, stacklevel=2)

    def save(self, filename: str) -> None:
        self.output(filename)
        Geometry.save(self, filename)

    def get_thermo_data(self, beta: float, N: int | None = None) -> ThermoData:
        data = ThermoData(beta, self.type_alias)
        data.energy = self.V
        self.trans(data, beta)
        self.rot(data, beta)
        self.vib(data, beta, N)
        return data

    def trans(self, data: ThermoData, beta: float) -> None:
        data.log_pf[0] = 3 * math.log(math.sqrt(sum(self.m) / (2 * np.pi * beta)) / HBAR) if self.n_zero >= 3 else 0

    def rot(self, data: ThermoData, beta: float, masses: float | NDArray = None) -> None:
        masses = self.masses if masses is None else masses
        if self.n_zero >= 5:
            pmi: NDArray = np.linalg.eigvalsh(inertia(self.x, masses))  # principal moments of inertia
            pmi = np.delete(pmi, np.isclose(pmi, 0))
            rot_const: NDArray = HBAR**2 / (2 * pmi)
            data.inertia = pmi
            data.rot_const = rot_const
            if self.n_zero == 5:
                data.log_pf[1] = -math.log(rot_const[1] * beta)
            else:
                data.log_pf[1] = 0.5 * math.log(np.pi / (math.prod(rot_const) * beta**3))

    def vib(self, data: ThermoData, beta: float, N: int | None = None) -> None:
        freqs: NDArray = self.freqs[self.n_zero + self.order :]
        if N is not None:
            freqs = 2 * N / (beta * HBAR) * np.arcsinh(beta * HBAR * freqs / (2 * N))
        if self.freqs.size > self.n_zero:
            data.freqs = np.sort(self.freqs[np.argpartition(abs(self.freqs), self.n_zero)[self.n_zero :]])
        else:
            data.freqs = np.array([])
        data.N = N
        data.log_pf[2] = -sum(np.log(2 * np.sinh(0.5 * beta * HBAR * freqs)))


@dataclass(slots=True)
class Minimum(StationaryPoint):
    second_mol: bool = False

    order: ClassVar[int] = 0
    type_alias: ClassVar[str] = "min"

    def __hash__(self) -> int:
        return -1 if self.second_mol else 0

    def update_links(self, *args) -> None:
        if len(args):
            raise ValueError("Minimum does not have links")


@dataclass(slots=True)
class TransitionState(StationaryPoint):
    order: ClassVar[int] = 1
    type_alias: ClassVar[str] = "ts"

    def __hash__(self) -> int:
        return 1

    def final_output(self, filename: str) -> None:
        StationaryPoint.final_output(self, filename)  # must explicitly call parent method if @dataclass(slots=True)
        beta_c = 2 * np.pi / (HBAR * (-self.freqs[0]))
        fmt = Formats.TEMPERATURE
        log.info(f"such that beta_c = {beta_c:{fmt}}, T_c = {Temperature.to_kelvin(beta_c):{fmt}} K")

    def get_inst_guess(self, N: int, beta: float, length: float = 0.1) -> "Instanton":
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
    """Springs for a full ring or its reflection-symmetric half."""

    N: int
    beta: float
    masses: NDArray
    omega_n: float = field(init=False)

    def __post_init__(self):
        self.omega_n: float = self.N / (self.beta * HBAR)

    def potential(self, x: NDArray) -> float:
        if len(x) == self.N:
            dx: NDArray = np.diff(x, axis=0, append=x[:1])
            return 0.5 * self.omega_n**2 * np.einsum("j,ijk,ijk", self.masses, dx, dx)
        elif len(x) * 2 == self.N:
            dx: NDArray = np.diff(x, axis=0)
            return self.omega_n**2 * np.einsum("j,ijk,ijk", self.masses, dx, dx)
        else:
            raise ValueError

    def gradient(self, x: NDArray) -> NDArray:
        if len(x) == self.N:
            return self.omega_n**2 * self.masses[:, None] * (2 * x - np.roll(x, 1, axis=0) - np.roll(x, -1, axis=0))
        if 2 * len(x) != self.N:
            raise ValueError("Invalid bead count")
        res: NDArray = np.zeros_like(x)
        dx: NDArray = np.diff(x, axis=0)
        res[:-1] -= dx
        res[1:] += dx
        return 2 * self.omega_n**2 * self.masses[:, None] * res

    def hessian(self, x: NDArray) -> NDArray:  # todo: banded
        if len(x) == self.N:
            return self.hessian_full(x)
        if 2 * len(x) != self.N:
            raise ValueError("Invalid bead count")
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
        res[indices, indices - d] -= tmp[indices % d]
        res[indices - d, indices] -= tmp[indices % d]
        return res


@dataclass(slots=True)
class Instanton(TransitionState):
    """Ring-polymer instanton; reflection-symmetric half ring by default."""

    beta: float | None = None
    full_ring: bool = field(default=False, kw_only=True)
    N: int = field(init=False)
    springs: Springs = field(init=False)
    type_alias: ClassVar[str] = "inst"

    def __post_init__(self):
        StationaryPoint.__post_init__(self)
        if self.beta is None:
            raise ValueError("beta must be specified for instanton")
        self.N: int = self.bead_weight * len(self.x)
        self.validate_nbeads(self.N)
        self.springs = Springs(self.N, self.beta, self.masses)

    def __setstate__(self, state):
        # Slot-based pickles written before full_ring existed have no such key.
        self.full_ring = False
        for name, value in state[1].items():
            setattr(self, name, value)

    @property
    def bead_weight(self) -> int:
        return 1 if self.full_ring else 2

    def validate_nbeads(self, N: int) -> None:
        if N < 2 or (not self.full_ring and N % 2):
            raise ValueError("Bead count must be >= 2 and even for a half ring")

    def to_full_ring(self) -> "Instanton":
        """Return an independent full-ring object, leaving this object unchanged.

        Mirror all bead-local data together. Full-ring spectra and sampling data
        remain valid; a half-ring optimizer's assembled Hessian cannot be expanded.
        """
        inst = deepcopy(self)
        if not inst.full_ring:
            for name in ("coords", "energy", "grad", "hess"):
                value = getattr(inst, name)
                if value is not None:
                    if name == "hess" and value.ndim == 2:
                        inst.hess = None
                    else:
                        setattr(inst, name, np.concatenate((value, value[::-1])))
            inst.full_ring = True
        return inst

    def __hash__(self) -> int:
        return 2

    @property
    def V(self) -> float:
        return self.bead_weight * sum(self.energy) + self.springs.potential(self.x)

    @V.setter
    def V(self, value: NDArray) -> None:
        self.energy = value

    @property
    def G(self) -> NDArray:
        return self.bead_weight * self.grad + self.springs.gradient(self.x)

    @G.setter
    def G(self, value: NDArray) -> None:
        self.grad = value

    def build_hess(self):
        res: NDArray = self.springs.hessian(self.x).reshape(len(self.x), self.dof, len(self.x), self.dof)
        indices: NDArray = np.arange(len(res))
        res[indices, :, indices, :] += self.bead_weight * self.hess
        return res.reshape(self.x.size, self.x.size)

    @property
    def H(self) -> NDArray:
        if self.hess.ndim == 3:
            return self.build_hess()
        else:  # ndim == 2
            return self.hess

    @H.setter
    def H(self, value: NDArray) -> None:
        self.hess = value

    def hessian_full(self) -> NDArray:
        inst = self if self.full_ring else self.to_full_ring()
        if inst.hess is None:
            raise ValueError("Full-ring Hessian requires bead Hessians; recompute them before analysis")
        return inst.H

    @property
    def dof(self) -> int:
        return self.x[0].size

    def interpolate(self, N: int) -> None:
        self.validate_nbeads(N)
        if isinstance(self, InstRef):
            center = self.x.mean(axis=0)
            if self.full_ring:
                self.x = CubicSpline(
                    np.arange(self.N + 1) / self.N, np.concatenate((self.x, self.x[:1])), bc_type="periodic"
                )(np.arange(N) / N)
            elif len(self.x) == 1:
                self.x = np.repeat(self.x, N // 2, axis=0)
            else:
                self.x = CubicSpline(np.linspace(0, 1, len(self.x)), self.x)(np.linspace(0, 1, N // 2))
            self.x += center - self.x.mean(axis=0)
            self.N = N
            self.springs = Springs(N, self.beta, self.masses)
            self.energy = self.grad = self.hess = self.freqs = self.modes = None
            self.harm_energies = None
            return
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

    def output(self, filename: str) -> None:
        if self.symbols is None:
            comment = f"V = {format_array(self.energy, fmt=Formats.ENERGY)}"
            np.savetxt(filename + ".txt", np.squeeze(self.x), fmt="%15.8f", header=comment)
        else:
            comment = [f"V = {V:{Formats.ENERGY}}" for V in self.energy] if self.V is not None else None
            save(filename + ".xyz", self.x, self.symbols, comment)

    def final_output(self, prefix: str) -> None:
        dx = np.diff(self.x, axis=0, append=self.x[:1]) if self.full_ring else np.diff(self.x, axis=0)
        contrib: NDArray = self.bead_weight * self.m * np.sum(dx**2, axis=(0, 2))
        BN: float = np.sum(contrib)
        fmt: str = Formats.BN
        log.info(f"mass-weighted BN: BN = {BN:{fmt}}, BN/(betaN*hbar) = {self.N * BN / (self.beta * HBAR):{fmt}}")
        if self.symbols is not None and BN > 0:
            log.info("Contributions to BN (squared mass-weighted path length) from various atoms:")
            for a, atom in enumerate(self.symbols):
                log.info(f"atom {a} ({atom}): {contrib[a] / BN:>5.1%}")
        log.info(f"S/hbar = {self.S / HBAR:{Formats.ACTION}}")
        self.save(prefix)

    @property
    def S(self) -> float:
        return self.beta / self.N * HBAR * self.V

    @property
    def E(self) -> float:
        """Tunneling energy"""
        return (self.bead_weight * sum(self.energy) - self.springs.potential(self.x)) / self.N

    @property
    def BN(self) -> float:
        dx = np.diff(self.x, axis=0, append=self.x[:1]) if self.full_ring else np.diff(self.x, axis=0)
        return self.bead_weight * np.einsum("j,ijk,ijk", self.m, dx, dx)

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
        lam: NDArray = np.linalg.eigvalsh(mass_weight(self.hessian_full(), self.m, dim=self.x.shape[-1]))
        self.freqs: NDArray = np.sqrt(abs(lam)) * np.sign(lam)
        freqs_nonzero: NDArray = self.freqs[np.argpartition(abs(self.freqs), self.n_zero + 1)[self.n_zero + 1 :]]
        order: int = sum(freqs_nonzero < 0)
        if order == 2 and Energy(freqs_nonzero[1], "au").get("cm-1") < 100:
            raise NotImplementedError
        elif order != 1:
            raise RuntimeError(f"Wrong number of imaginary frequencies (expected 1, got {order} instead)")
        beta_n: float = beta / self.N
        res = -sum(np.log(beta_n * HBAR * abs(freqs_nonzero))) + (self.n_zero + 1) * math.log(self.N)
        res += 0.5 * (math.log(2 * np.pi * BN) - math.log(beta_n * HBAR**2))
        data.freqs = np.sort(freqs_nonzero)
        data.log_pf[2] = res


@dataclass(slots=True)
class HarmRef(Geometry):
    N: int | None = field(init=False, default=None)
    T: float | None = field(init=False, default=None)
    harm_energies: NDArray | None = field(init=False, default=None)

    def update_links(self, *args) -> None:
        if len(args):
            raise ValueError("HarmRef does not have links")

    def calc_freq(self) -> None:
        Geometry.calc_freq(self)
        self._norm_dimensionless_modes()

    def _norm_dimensionless_modes(self) -> None:
        modes_raw = self.modes.T.reshape(self.dof, len(self.m), 3)  # shape (3N, N, 3)
        mass_amu = self.m * Mass(1, "au").get("amu")
        mass_factor = mass_amu[np.newaxis, :, np.newaxis] ** -0.5  # shape (1, N, 1)
        self.modes = modes_raw * mass_factor

    def get_inst_guess(self, N: int, beta: float, length: float = 0.1, *, full_ring: bool = False) -> "InstRef":
        if N is None or N < 2 or (not full_ring and N % 2):
            raise ValueError("Specify a bead count >= 2 (even for a half ring)")
        if self.modes is None:
            self.calc_freq()
        mode = self.modes[0] / norm(self.modes[0])
        phase = 2 * np.pi * np.arange(N) / N if full_ring else np.linspace(0, math.pi, N // 2)
        x_inst: NDArray = self.x + length * mode[None, ...] * np.cos(phase).reshape(-1, *(1,) * self.x.ndim)
        x_inst += self.x - x_inst.mean(axis=0)
        return InstRef(x_inst, self.symbols, links=[self], masses=self.m, beta=beta, full_ring=full_ring)

    def delta_free_energy(self) -> float:
        freqs_complex = np.where(self.freqs > 0, 1, -1j) * self.freqs
        beta = 1.0 / (self.T * KB)
        hbfs = 0.5 * beta * freqs_complex
        return (np.sum(np.log(np.sinh(np.arcsinh(hbfs / self.N) * self.N)) - np.log(hbfs)) / beta).real


@dataclass(slots=True)
class InstRef(Instanton):
    T: float | None = field(init=False, default=None)
    harm_energies: NDArray | None = field(init=False, default=None)

    order: ClassVar[int] = 0
    type_alias: ClassVar[str] = "centroid"

    def __post_init__(self):
        Instanton.__post_init__(self)
        self.n_zero = self.x[0].size
        self.T = 1 / (KB * self.beta)

    def validate_reference(self) -> None:
        if len(self.links) != 1 or not isinstance(self.links[0], HarmRef):
            raise ValueError("InstRef FEP requires one HarmRef link")
        ref = self.links[0]
        if (
            not np.array_equal(self.symbols, ref.symbols)
            or self.m.shape != ref.m.shape
            or not np.allclose(self.m, ref.m)
            or self.x.shape[1:] != ref.x.shape
            or not np.allclose(self.x.mean(axis=0), ref.x, rtol=0, atol=1e-7)
        ):
            raise ValueError("HarmRef symbols, masses and centroid must match the instanton")

    def update_links(self, *args) -> None:
        if len(args) < 2:
            self.links = list(args)
        else:
            raise ValueError("InstRef only accept one link")

    @property
    def G(self) -> NDArray:
        res: NDArray = self.bead_weight * self.grad + self.springs.gradient(self.x)
        return res - np.mean(res, axis=0)

    @G.setter
    def G(self, value: NDArray) -> None:
        self.grad = value

    @property
    def H(self) -> NDArray:
        p = centroid(self.x).reshape(-1, self.x.size)
        p_mat = np.identity(self.x.size) - np.einsum("ij,ik->jk", p, p)
        hess = Instanton.build_hess(self) if self.hess.ndim == 3 else self.hess
        return p_mat @ hess @ p_mat

    @H.setter
    def H(self, value: NDArray) -> None:
        self.hess = value

    def set_beta(self, beta: float) -> None:
        Instanton.set_beta(self, beta)
        self.T = 1 / (KB * beta)

    def final_output(self, prefix: str) -> None:
        self.harm_energies = None
        hess_mw = mass_weight(self.hessian_full(), self.m, dim=self.x.shape[-1])
        eigs, self.modes = np.linalg.eigh(hess_mw)
        self.freqs = np.sqrt(abs(eigs)) * np.sign(eigs)
        Instanton.final_output(self, prefix)

    def delta_free_energy(self) -> float:
        self.validate_reference()
        if np.isclose(BN := self.BN, 0):
            df: float = self.x[0].size * np.log(self.N)
        else:
            df = (self.x[0].size + 1) * np.log(self.N) + 0.5 * np.log(BN * self.N / (2 * np.pi * self.beta * HBAR**2))
        df = -(df - sum(np.log(self.beta / self.N * self.freqs))) / self.beta
        df += np.mean(self.energy) + self.springs.potential(self.x) / self.N - self.links[0].energy
        return df
