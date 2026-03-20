import inspect
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import IntEnum
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from pyrinst.utils.elements import element_data
from pyrinst.utils.units import Mass

if TYPE_CHECKING:
    from pyrinst.geometries import Geometry

POTENTIAL_REGISTRY: dict[str, type["Potential"]] = {}


class Task(IntEnum):
    SP = 0  # Single Point
    GRAD = 1  # Gradient
    FREQ = 2  # Frequency


class Potential(ABC):
    """
    Base class upon which all other potentials should be based. ABC is an abstract class that does not allow
    instantiation. By default, hessian is computed by finite differences.
    """

    type_alias: str | None = None
    dx: float = 1e-4

    def __init_subclass__(cls):
        if not inspect.isabstract(cls):
            POTENTIAL_REGISTRY[cls.type_alias or cls.__name__.lower()] = cls

    @abstractmethod
    def __call__(self, x: NDArray, task: Task = Task.GRAD) -> tuple[float, NDArray | None, NDArray | None]: ...

    def compute(self, geom: "Geometry", task: Task = Task.GRAD) -> None:
        """
        Compute the potential-energy surface properties.

        The attributes `energy`, `grad`, and optionally `hess` of the input `geometry` object will be updated in-place
        upon successful completion.

        Parameters
        ----------
        geom : Geometry
            The data container holding atomic coordinates and symbols.
        task : Task, optional
            The task to perform, by default Task.GRAD.
        """
        if geom.x.ndim == 3:
            func: Callable[[NDArray], tuple[float, NDArray | None, NDArray | None]] = partial(self, task=task)
            res = [np.array([res[i] for res in tuple(map(func, geom.x))]) for i in range(3)]
        else:
            res = self(geom.x, task)
        geom.V = res[0]
        geom.G = res[1]
        geom.H = res[2]


# assumes that we are using some electronic structure program
# which we call through an executable with some input that files we generate
# and that gives back some output files which we parse


class OnTheFlyDriver(Potential, ABC):
    _runcmd: str = None
    _args: str = ""

    def __init__(self, symbols: list[str], template_input: str, runcmd: str = None, working_dir: str = ".", **_):
        self.symbols = element_data.get_base_symbols(symbols)
        self._template_input = template_input
        self._runcmd: str = runcmd or os.environ.get("RUNCMD") or self._runcmd
        self._sys_name: str = self.__class__.__name__.lower()
        self._working_dir = os.path.abspath(working_dir)
        self._max_tracker: int | None = None
        self._folder: str = ""
        if self._working_dir != os.getcwd():
            shutil.rmtree(self._working_dir, ignore_errors=True)
        os.makedirs(self._working_dir, exist_ok=True)
        self._tracker: int = 0
        self._super_tracker: int = 0
        self._cached: [dict] = [{}, {}, {}]

    def __call__(self, x: NDArray, task: Task = Task.GRAD) -> tuple[float, NDArray | None, NDArray | None]:
        hash_x = hash(x.tobytes())
        if self._cached[task].get(hash_x, None) is not None:
            ans = [None] * 3
            for i in range(task + 1):
                ans[i] = self._cached[i][hash_x]
            return tuple(ans)
        self._update_tracker()
        self.generate_input(x, task)
        runcmd: str = f"cd {self._folder}; {self._runcmd} {self._args}"
        subprocess.run(runcmd, capture_output=True, check=True, shell=True)
        res = self.parse_output()
        self._cached[0][hash_x] = res.energy
        self._cached[1][hash_x] = res.grad
        self._cached[2][hash_x] = res.hess
        return res.energy, res.grad, res.hess

    def compute(self, geom: "Geometry", task: Task = Task.GRAD):
        if geom.x.ndim == 3:
            self._max_tracker = len(geom.x)
        super().compute(geom, task)

    def _update_tracker(self):
        if self._max_tracker is None:
            self._folder = f"{self._working_dir}/{self._tracker}"
        else:
            if self._tracker >= self._max_tracker:
                self._super_tracker += 1
                self._tracker = 0
            self._folder = f"{self._working_dir}/{self._super_tracker}/{self._tracker}"
        self._tracker += 1
        os.makedirs(self._folder)

    @abstractmethod
    def generate_input(self, x: NDArray, task: Task = Task.GRAD):
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
