import numpy as np
from numpy.typing import NDArray

from pyrinst.io.xyz import lines
from pyrinst.utils.elements import element_data

from .base import OnTheFlyDriver, OnTheFlyResult, Task


class Orca(OnTheFlyDriver):
    _runcmd: str = "orca"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._grad_cmd: str = "! EnGrad\n"
        self._hess_cmd: str = "! Freq\n"
        self._input: str = f"{self._sys_name}.inp"
        self._output: str = f"{self._sys_name}.out"
        self._args: str = f"{self._input} > {self._output}"
        self.main_input: str = ""
        self.add_input: str = ""
        with open(self._template_input) as f:
            for line in f:
                if line.startswith("!"):
                    self.main_input += line
                else:
                    self.add_input += line

    def generate_input(self, x: NDArray, task: Task = Task.GRAD):
        input_file: str = f"{self._folder}/{self._input}"
        assert len(x) == len(self.symbols)
        with open(input_file, "w") as f:
            f.write(self.main_input)
            if task == Task.FREQ:
                f.write(self._hess_cmd)
            if task >= Task.GRAD:
                f.write(self._grad_cmd)
            f.write(self.add_input + lines(self.symbols, x) + "*\n")

    def parse_output(self):
        prefix: str = f"{self._folder}/{self._sys_name}"
        return OrcaResult(prefix)


class OrcaResult(OnTheFlyResult):
    def read(self, prefix):
        with open(f"{prefix}.engrad") as f:
            lines = [line.strip() for line in f if "#" not in line and line.strip()]
        n_atoms, self.energy = int(lines[0]), float(lines[1])
        self.grad = np.array(lines[2 : 2 + 3 * n_atoms], dtype=float).reshape(-1, 3)
        self.coord = np.array([line.split() for line in lines[2 + 3 * n_atoms :]], dtype=float).reshape(-1, 4)
        self.symbols = element_data.get_symbols(np.astype(self.coord[:, 0], int))
        self.coord = self.coord[:, 1:]
        try:
            with open(prefix + ".hess") as f:
                for line in f:
                    if "$hessian" in line:
                        self._read_hess(f, line)
                    elif "$atoms" in line:
                        self._read_mass(f, line)
        except FileNotFoundError:
            pass

    def _read_hess(self, f, line):
        dof: int = int(next(f).strip())
        self.hess = np.zeros((dof, dof))
        for i in range(dof // 5):
            next(f)
            for j in range(dof):
                self.hess[j, i * 5 : min((i + 1) * 5, dof)] = [float(d) for d in next(f).split()[1:]]

    def _read_mass(self, f, line):
        n_atom: int = int(next(f).strip())
        self.mass = np.zeros(n_atom)
        for i in range(n_atom):
            self.mass[i] = float(next(f).split()[1])
