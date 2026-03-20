import logging
import math
import re

import numpy as np
from numpy.typing import NDArray

from pyrinst.io.xyz import lines
from pyrinst.utils.elements import element_data

from .base import OnTheFlyDriver, SingleFileResult, Task

log = logging.getLogger(__name__)


class Gaussian(OnTheFlyDriver):  # todo: not fully tested
    _runcmd: str = "g09"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._grad_cmd: str = "Force"
        self._hess_cmd: str = "Freq"
        self._link: int = 0
        self._header: list[str] = [""]
        self._tail: list[str] = [""]
        self._charge_mult: str = ""
        flag: int = 1  # 1: header, 2: title, 3: charge/mult, 4: tail
        with open(self._template_input) as f:
            for line in f.readlines():
                if line.strip() == "":
                    flag += 1
                    continue
                match flag:
                    case 1:  # header
                        if "chk" in line.lower():
                            self._sys_name = line.split("=")[1].split(".")[0]  # should be the same name for all links
                        elif "force" in line.lower() or "freq" in line.lower():
                            line = re.sub("force|freq", "", line, flags=re.IGNORECASE)
                        self._header[self._link] += line
                    case 2:  # title
                        if "nosymm" not in self._header[self._link].lower():
                            log.info("Adding 'NoSymm' keyword to input to prevent Gaussian from rotating the molecule")
                            self._header[self._link] += "NoSymm\n"
                        self._title: str = line
                    case 3:  # charge and multiplicity
                        if self._charge_mult == "":  # otherwise skip geometry
                            self._charge_mult: str = line  # should be the same name for all links
                    case 4:  # additional options
                        self._tail[self._link] += line
                        if "--link1--" in line.lower():
                            self._link += 1
                            self._header.append("")
                            self._tail.append("")
                            flag = 1
        self._link += 1
        self._input: str = f"{self._sys_name}.com"
        self._output: str = f"{self._sys_name}.log"
        self._args: str = f"< {self._input} > {self._output} ; formchk {self._sys_name}.chk"

    def generate_input(self, x: NDArray, task: Task = Task.GRAD):
        input_file: str = f"{self._folder}/{self._input}"
        assert len(x) == len(self.atoms)
        with open(input_file, "w") as f:
            for i in range(self._link):
                f.write(self._header[i])
                if i == self._link - 1:
                    if task == Task.FREQ:
                        f.write(f"{self._hess_cmd}\n")
                    elif task == Task.GRAD:
                        f.write(f"{self._grad_cmd}\n")
                f.write(f"\n{self._title}\n{self._charge_mult}{lines(self.atoms, x)}\n\n{self._tail[i]}\n")

    def parse_output(self):
        prefix: str = f"{self._folder}/{self._sys_name}"
        return GaussianResult(prefix)


class GaussianResult(SingleFileResult):
    ext = ".fchk"
    _patterns = (
        "Atomic numbers",  # read atomic numbers before coordinates
        "Total Energy",
        "Cartesian Gradient",
        "Cartesian Force Constants",
    )

    def _read_coord(self, f, line):
        """Reads coordinates from the .fchk file. The atomic numbers are read first. The coordinates are read next.
        The masses are read last.
        """
        # read atomic numbers
        atom_num = []
        for _ in range(math.ceil(int(line.split()[-1]) / 6.0)):
            atom_num += f.readline().split()
        self.atoms = element_data.get_symbols(np.array(atom_num, dtype=int))
        # read coordinates
        for line in f:
            if "Current cartesian coordinates" in line:
                break
        coord = []
        for _ in range(math.ceil(int(line.split()[-1]) / 5.0)):
            coord += f.readline().split()
        self.coord = np.array(coord, dtype=float).reshape((-1, 3))
        # read masses
        for line in f:
            if "Real atomic weights" in line:
                break
        mass = []
        for _ in range(math.ceil(int(line.split()[-1]) / 5.0)):
            mass += f.readline().split()
        self.mass = np.array(mass, dtype=float)

    def _read_energy(self, f, line):
        self.energy = float(line.split()[3])

    def _read_grad(self, f, line):
        dim = int(line.split()[-1])
        self.grad = np.zeros(dim)
        for i in range(dim // 5):
            self.grad[i * 5 : (i + 1) * 5] = np.array([float(d) for d in next(f).split()])
        if dim % 5 > 0:
            self.grad[dim // 5 * 5 :] = np.array([float(d) for d in next(f).split()])
        self.grad = self.grad.reshape((dim // 3, 3))

    def _read_hess(self, f, line):
        nh = int(line.split()[-1])
        hess_ar = np.zeros(nh)
        for i in range(nh // 5):
            hess_ar[i * 5 : (i + 1) * 5] = np.array([float(d) for d in next(f).split()])
        if nh % 5 > 0:
            hess_ar[nh // 5 * 5 :] = np.array([float(d) for d in next(f).split()])

        dim = int((np.sqrt(1 + 8 * nh) - 1) / 2)
        self.hess = np.zeros([dim, dim])
        for i in range(dim):
            for j in range(i + 1):
                self.hess[i, j] = self.hess[j, i] = hess_ar[i * (i + 1) // 2 + j]
