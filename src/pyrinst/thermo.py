import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .io.formats import Formats, format_array
from .utils.units import Energy, Length, Time

if TYPE_CHECKING:
    from .geometries import StationaryPoint

log = logging.getLogger(__name__)
hbar: float = 1


@dataclass(slots=True)
class ThermoData:
    beta: float
    data_type: str
    energy: float | None = field(default=None, init=False)
    log_pf: NDArray = field(default_factory=lambda: np.zeros(3), init=False)
    inertia: NDArray | None = field(default=None, init=False)
    rot_const: NDArray | None = field(default=None, init=False)
    freqs: NDArray | None = field(default=None, init=False)
    N: int | None = field(default=None, init=False)
    unimolecular: bool = field(default=True, init=False)

    def __add__(self, other):
        if self.data_type != "min" or other.data_type != "min":
            raise ValueError("Only Minima can be added together!")
        res = ThermoData(self.beta, self.data_type)
        res.energy = self.energy + other.energy
        res.log_pf = self.log_pf + other.log_pf
        res.unimolecular = False
        return res


class ThermoRate(ABC):
    title: str = ""
    data_name1: str = ""
    data_name2: str = ""

    def __init__(self, data1: ThermoData, data2: ThermoData) -> None:
        self.data1 = data1
        self.data2 = data2
        if data1.unimolecular and not math.isclose(data1.log_pf[0], data2.log_pf[0]):
            raise ValueError("Ratio between Z_trans should be 1 for a unimolecular reaction!")
        if not math.isclose(data1.beta, data2.beta):
            raise ValueError("Betas must be equal!")
        self.beta: float = data1.beta
        self.barrier: float = data2.energy - data1.energy
        self.log_pf: NDArray = data2.log_pf - data1.log_pf

    def show(self) -> None:
        log.info(f"\nComputing {self.title} from {self.data_name2} and {self.data_name1}...")
        log.info(f"Partition-function ratios ({self.data_name2}/{self.data_name1}):")
        pf: NDArray = np.exp(self.log_pf)
        fmt: str = Formats.PARTITION_FUNCTION
        log.info(f"  trans {pf[0]:{fmt}}\n  rot   {pf[1]:{fmt}}\n  vib   {pf[2]:{fmt}}")
        rate: float = self.get_rate()
        self.show_barrier()
        self.show_rate(rate)

    @abstractmethod
    def get_rate(self) -> float: ...

    @abstractmethod
    def show_barrier(self) -> None: ...

    def show_rate(self, rate: float) -> None:
        if self.data1.unimolecular:
            rate_si: float = rate * Time(1, "au").get("s") ** -1
            log.info(f"k({self.__class__.__name__}) = {rate:{Formats.RATE}} = {rate_si:{Formats.RATE}} s^-1")
            log.info(f"log10(k({self.__class__.__name__}) / s^-1) = {math.log10(rate_si):{Formats.LOG_RATE}}")
        else:
            rate_si: float = rate * Length(1, "au").get("cm") ** 3 / Time(1, "au").get("s")
            log.info(f"k({self.__class__.__name__}) = {rate:{Formats.RATE}} = {rate_si:{Formats.RATE}} cm^3 s^-1")
            log.info(f"log10(k({self.__class__.__name__}) / cm^3 s^-1) = {math.log10(rate_si):{Formats.LOG_RATE}}")


class Eyring(ThermoRate):
    title: str = "Eyring TST rate"
    data_name1: str = "reactant"
    data_name2: str = "TS"

    def get_rate(self) -> float:
        return math.exp(sum(self.log_pf) - self.beta * self.barrier) / (2 * np.pi * self.beta * hbar)

    def show_barrier(self) -> None:
        log.info(f"Barrier = {self.barrier:{Formats.ENERGY}}")


class PB(Eyring):  # Exact rate for a parabolic barrier
    def show_rate(self, rate: float) -> None:
        freq: float = -self.data2.freqs[0]
        k_pb: float = rate * 0.5 * self.beta * hbar * freq / math.sin(0.5 * self.beta * hbar * freq)
        super().show_rate(rate)
        if self.data1.unimolecular:
            rate_si: float = k_pb / Time(1, "au").get("s")
            log.info(f"k(PB) = {k_pb:{Formats.RATE}} = {rate_si:{Formats.RATE}} s^-1")
            log.info(f"log10(k(PB) / s^-1) = {math.log10(rate_si):{Formats.LOG_RATE}}")
        else:
            rate_si: float = k_pb * Length(1, "au").get("cm") ** 3 / Time(1, "au").get("s")
            log.info(f"k(PB) = {k_pb:{Formats.RATE}} = {rate_si:{Formats.RATE}} cm^3 s^-1")
            log.info(f"log10(k(PB) / cm^3 s^-1) = {math.log10(rate_si):{Formats.LOG_RATE}}")


class Inst(Eyring):
    title: str = "thermal instanton rate"
    data_name1: str = "reactant"
    data_name2: str = "instanton"

    def show_barrier(self) -> None:
        log.info(f"S/hbar - beta*V(r) = {self.beta * self.barrier:{Formats.ACTION}}")


class TunnelingFactor(ThermoRate):
    title: str = "instanton tunneling factor"
    data_name1: str = "TS"
    data_name2: str = "instanton"

    def show_barrier(self) -> None:
        log.info(f"S/hbar - beta*V(TS) = {self.beta * self.barrier:{Formats.ACTION}}")

    def get_rate(self) -> float:
        return math.exp(sum(self.log_pf) - self.beta * self.barrier)

    def show_rate(self, rate: float) -> None:
        log.info(f"k(Inst)/k(Eyring) = {rate:{Formats.TUNNELING_FACTOR}}")


def analyze(sp: "StationaryPoint", beta: float) -> None:
    try:
        N: int | None = sp.N
    except AttributeError:
        N = None
    data: list[ThermoData] = [arg.get_thermo_data(beta, N) for arg in sp.links]
    for d in data:
        name: str = d.data_type
        log.info(f"\n{'-' * (len(name) + 2)}\n {name}\n{'-' * (len(name) + 2)}")
        log.info(f"V = {d.energy:{Formats.ENERGY}}")
        log.info(f"Z_trans = {math.exp(d.log_pf[0]):{Formats.PARTITION_FUNCTION}} per volume")
        if d.inertia is not None:
            log.info(f"Moments of Inertia = {format_array(d.inertia, fmt=Formats.MOMENT_OF_INERTIA)}")
            log.info(f"Rotational Constants = {format_array(d.rot_const, fmt=Formats.ROTATIONAL_CONSTANT)}")
        log.info(f"Z_rot = {math.exp(d.log_pf[1]):{Formats.PARTITION_FUNCTION}}")
        zpe = 0.5 * hbar * np.sum(d.freqs, where=d.freqs > 0)
        freqs_cm: NDArray = d.freqs[:12] * hbar * Energy(1, "au").get("cm-1")
        log.info(f"frequencies in cm-1:\n{format_array(freqs_cm, fmt=Formats.FREQUENCY)}")
        if name != "inst":
            log.info(f"H.O. ZPE = {Energy(zpe, 'au').get('cm-1'):{Formats.FREQUENCY}} cm-1")
        order: int = 0 if d.data_type == "min" else 1
        if (n := sum(d.freqs < 0)) != order:
            raise ValueError(f"Wrong number of negative eigenvalues (expected {order}, got {n} instead)")
        msg: str = "(exact harmonic)" if d.N is None else f"({d.N}-bead approx)"
        log.info(f"{msg} log(Z_vib) = {d.log_pf[2]:{Formats.LOG_PARTITION_FUNCTION}}")

    try:
        tmp: ThermoData = data[0] + data[1]  # combine two minima
        data = [tmp] + data[2:]
    except ValueError:  # unimolecular reaction
        pass
    except IndexError:  # only one stationary point
        return

    for d2 in data[1:]:
        for d1 in data[1::-1]:
            if d1.data_type == "min" and d2.data_type == "ts":
                Eyring(d1, d2).show()
                if beta < 2 * np.pi / (hbar * (-d2.freqs[0])):
                    PB(d1, d2).show()
            elif d1.data_type == "min" and d2.data_type == "inst":
                Inst(d1, d2).show()
            elif d1.data_type == "ts" and d2.data_type == "inst":
                TunnelingFactor(d1, d2).show()
