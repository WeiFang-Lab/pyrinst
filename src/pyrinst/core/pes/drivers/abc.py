import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable

from numpy.typing import NDArray

from pyrinst.utils.elements import element_data
from pyrinst.utils.units import Mass

# assumes that we are using some electronic structure program
# which we call through an executable with some input that files we generate
# and that gives back some output files which we parse


class OnTheFlyDriver(ABC):
    _runcmd: str = None
    _args: str = ""

    def __init__(self, atoms: list[str], template_input: str, runcmd: str = None, working_dir: str = ".", **_):
        self.atoms = element_data.get_base_symbols(atoms)
        self._template_input = template_input
        self._runcmd: str = runcmd or os.environ.get("RUNCMD") or self._runcmd
        self._sys_name: str = self.__class__.__name__.lower()
        self._working_dir = os.path.abspath(working_dir)
        self._max_tracker: int | None = None
        self._folder: str = ""
        self._prepare_working_dir()

    def _prepare_working_dir(self):
        if self._working_dir != os.getcwd():
            shutil.rmtree(self._working_dir, ignore_errors=True)
        os.makedirs(self._working_dir, exist_ok=True)
        self._tracker: int = -1
        self._super_tracker: int = 0

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._prepare_working_dir()

    def set_max_tracker(self, max_tracker: int):
        self._max_tracker = max_tracker
        self._folder = f"{self._working_dir}/{self._super_tracker}/{self._tracker}"

    def _update_tracker(self):
        self._tracker += 1
        if self._max_tracker is None:
            self._folder = f"{self._working_dir}/{self._tracker}"
        else:
            breakpoint()
            if self._tracker >= self._max_tracker:
                self._super_tracker += 1
                self._tracker = 0
            self._folder = f"{self._working_dir}/{self._super_tracker}/{self._tracker}"
        os.makedirs(self._folder)

    @abstractmethod
    def generate_input(self, x: NDArray, calc_grad: bool = True, calc_hess: bool = True):
        """Generate input files for the electronic structure calculation."""

    @abstractmethod
    def parse_output(self):
        """Parse the output of the electronic structure calculation."""

    def compute(self, x: NDArray, calc_grad: bool = True, calc_hess: bool = True):
        self._update_tracker()
        self.generate_input(x, calc_grad, calc_hess)
        runcmd: str = f"cd {self._folder}; {self._runcmd} {self._args}"
        subprocess.run(runcmd, capture_output=True, check=True, shell=True)
        return self.parse_output()


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
        self.atoms = None
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
        with open(prefix + self.ext, "r") as f:
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
