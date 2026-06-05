import inspect
import os
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import IntEnum

from numpy.typing import NDArray

from pyrinst.utils.elements import element_data
from pyrinst.utils.units import Mass

POTENTIAL_REGISTRY: dict[str, type["Potential"]] = {}


class Level(IntEnum):
    ENER = 0  # Energy
    GRAD = 1  # Gradient
    FREQ = 2  # Frequency


PotentialResult = tuple[float, NDArray | None, NDArray | None]
BatchPotentialResult = tuple[NDArray, NDArray | None, NDArray | None]


class Potential(ABC):
    """
    Base class for a single-geometry calculator.
    """

    type_alias: str | None = None
    dx: float = 1e-4

    def __init_subclass__(cls):
        if not inspect.isabstract(cls):
            POTENTIAL_REGISTRY[cls.type_alias or cls.__name__.lower()] = cls

    @abstractmethod
    def __call__(self, x: NDArray, level: Level = Level.GRAD) -> PotentialResult: ...


# assumes that we are using some electronic structure program
# which we call through an executable with some input that files we generate
# and that gives back some output files which we parse


class OnTheFlyPotential(Potential, ABC):
    _runcmd: str = None
    _args: str = ""

    def __init__(self, symbols: list[str], template_input: str, runcmd: str = None, working_dir: str = ".", **_):
        self.symbols = element_data.get_base_symbols(symbols)
        self._template_input = template_input
        self._runcmd: str = runcmd or os.environ.get("RUNCMD") or self._runcmd
        self._sys_name: str = self.__class__.__name__.lower()
        self._working_dir = os.path.abspath(working_dir)
        self._folder: str = self._working_dir
        os.makedirs(self._working_dir, exist_ok=True)

    def __call__(self, x: NDArray, level: Level = Level.GRAD) -> PotentialResult:
        self.generate_input(x, level)
        self.run()
        res = self.parse_output()
        return res.energy, res.grad, res.hess

    def run(self) -> None:
        runcmd: str = f"{self._runcmd} {self._args}"
        subprocess.run(runcmd, cwd=self._folder, capture_output=True, check=True, shell=True)

    @abstractmethod
    def generate_input(self, x: NDArray, level: Level = Level.GRAD):
        """Generate input files for the electronic structure calculation."""

    @abstractmethod
    def parse_output(self):
        """Parse the output of the electronic structure calculation."""


class OnTheFlyResult(ABC):
    """
    Abstract class that reads the results from the output file(s).

    Parameters
    ----------
    prefix : str
        The prefix of the output file(s).
    """

    length_scale: float = 1.0
    energy_scale: float = 1.0

    def __init__(self, prefix):
        self.coord = None
        self.energy = None
        self.grad = None
        self.hess = None
        self.symbols = None
        self.mass = None
        self.read(prefix)
        self.convert_units()

    @abstractmethod
    def read(self, prefix):
        """
        Read all the attributes if found from the output file(s).

        Parameters
        ----------
        prefix : str
            The prefix of the output file(s).
        """

    def convert_units(self):
        """
        Convert the units of the attributes to the new units.
        """
        self.coord *= self.length_scale
        self.energy *= self.energy_scale
        if self.mass is not None:
            self.mass *= Mass(1, "amu").get("au")
        if self.grad is not None:
            self.grad *= self.energy_scale / self.length_scale
        if self.hess is not None:
            self.hess *= self.energy_scale / self.length_scale**2


class SingleFileResult(OnTheFlyResult):
    """Abstract class that reads results from a single file, like Gaussian."""

    ext = ".out"  # file extension
    _patterns = (  # override it to match the output file
        "coordinates",
        "energy",
        "gradient",
        "hessian",
    )

    def __init__(self, prefix):
        self._readers: tuple[Callable, ...] = (self._read_coord, self._read_energy, self._read_grad, self._read_hess)
        super().__init__(prefix)

    def read(self, prefix):
        with open(prefix + self.ext) as f:
            self._read_preamble(f)
            for line in f:
                for i, pattern in enumerate(self._patterns):
                    if pattern in line:
                        self._readers[i](f, line)
                        break

    def _read_preamble(self, f):
        """
        Read information from the preamble if needed.

        Parameters
        ----------
        f : TextIOWrapper
            File object that is reading the output file.
        """

    @abstractmethod
    def _read_coord(self, f, line):
        """
        Parameters
        ----------
        f : TextIOWrapper
            File object that is reading the output file.
        line : str
            The last line has been read.
        """

    @abstractmethod
    def _read_energy(self, f, line):
        """
        Parameters
        ----------
        f : TextIOWrapper
            File object that is reading the output file.
        line : str
            The last line has been read.
        """

    @abstractmethod
    def _read_grad(self, f, line):
        """
        Parameters
        ----------
        f : TextIOWrapper
            File object that is reading the output file.
        line : str
            The last line has been read.
        """

    @abstractmethod
    def _read_hess(self, f, line):
        """
        Parameters
        ----------
        f : TextIOWrapper
            File object that is reading the output file.
        line : str
            The last line has been read.
        """
