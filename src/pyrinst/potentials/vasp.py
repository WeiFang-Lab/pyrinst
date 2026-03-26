import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from pyrinst.utils.units import ANGSTROM, EV, Energy, Length, Mass

from .base import OnTheFlyDriver, OnTheFlyResult, Task


class Vasp(OnTheFlyDriver):
    def __init__(self, *args, cell, add_files: Sequence[str], **kwargs):
        super().__init__(*args, **kwargs)
        if len(cell) == 3:
            self._cell = np.diag(cell)
        elif len(cell) == 9:
            self._cell = np.array(cell).reshape(3, 3)
        else:
            raise ValueError("Invalid unit cell input")
        self._hess_cmd: str = "IBRION=5; POTIM=0.01; NFREE=2\n"

        # Process additional input options
        self._add_files: dict[str, str] = {}
        for file in add_files:
            with open(file, "rb" if file == "vdw_kernel.bindat" else "r") as f:
                self._add_files[file] = f.read()
        with open(self._template_input) as f:
            self.incar = f.read()

        # use POTCAR path if POSCAR not provided; check KPOINTS
        if "POTCAR" not in self._add_files:
            print("POTCAR file not provided,", end=" ")
            print("trying to generate POTCAR from vasp potcar lib specified in env variable VASP_PP_PATH")
            self._gen_potcar(os.environ.get("VASP_PP_PATH"))
        if "KPOINTS" not in self._add_files:
            raise RuntimeError("KPOINTS file not provided")
        self.cwd = os.getcwd()
        self._output: str = f"{self._sys_name}.out"
        self._args = f" > {self._output}"

    def compute(self, geom, task: Task = Task.GRAD):
        super().compute(geom, task)
        self.cleanup()

    def _gen_potcar(self, vasp_pp_path):
        """Writes out the POTCAR with elements in the correct order
        An example: for 'CH3COOH', this should write the POTCAR with elements C H C O H
        The ase routine always sorts the element order and will cause trouble for us when we get forces.
        """
        if vasp_pp_path is None:
            raise RuntimeError("VASP_PP_PATH not specified. Either specified this or give a POTCAR file")

    def generate_input(self, x: NDArray, task: Task = Task.GRAD):
        """Sets up the system for each VASP calculation."""
        os.chdir(self._folder)
        with open("INCAR", "w") as f:
            if task == Task.FREQ:
                f.write("\n" + self._hess_command)
            f.write(self.incar)
        self._gen_poscar(x)

        for file, content in self._add_files.items():
            with open(file, "wb" if file == "vdw_kernel.bindat" else "w") as f:
                f.write(content)
        os.chdir(self.cwd)

    def _gen_poscar(self, x: NDArray):
        """Writes out the POSCAR file for a given configuration.
        'x' is a N x 3 np array of the coordinates of the atoms that are allowed to relax.
        """
        counter = defaultdict(int)
        for element in self.symbols:
            counter[element] += 1
        with open("POSCAR", "w") as f:
            f.write("POSCAR\n   1.0\n")
            for i in range(3):
                for j in range(3):
                    f.write(f"{self._cell[i, j]:>22.16f}")
                f.write("\n")
            f.write("   ")
            f.write("    ".join(counter.keys()) + "\n     ")
            f.write("    ".join(map(str, counter.values())) + "\n")
            f.write("Cartesian\n")
            for i in range(len(self.symbols)):
                for j in range(3):
                    f.write(f"{x[i, j]:>21.16f}")
                f.write("\n")

    def parse_output(self):
        return VaspResult(self._folder)

    def cleanup(self):
        """Deletes all wave functions and charges files when all calculations finish.
        These WAVECARs takes up a lot of space and quite worthless here.
        """
        os.chdir(self._working_dir)
        for f in os.walk("./").next()[1]:
            if os.path.exists(f + "/WAVECAR"):
                os.remove(f + "/WAVECAR")
            if os.path.exists(f + "/CHGCAR"):
                os.remove(f + "/CHGCAR")
            if os.path.exists(f + "/CHG"):
                os.remove(f + "/CHG")
        os.chdir(self.cwd)


class VaspResult(OnTheFlyResult):
    length_scale: float = ANGSTROM
    energy_scale: float = EV

    _ibrion_path = './/parameters/separator[@name="ionic"]/i[@name="IBRION"]'
    _coord_path_in = './/structure[@name="initialpos"]/varray[@name="positions"]/v'
    _coord_path_out = './/structure[@name="finalpos"]/varray[@name="positions"]/v'
    _select_path = './/structure[@name="finalpos"]/varray[@name="selective"]/v'
    _cell_path = './/structure[@name="finalpos"]/crystal/varray[@name="basis"]/v'
    _energy_path = './/calculation/energy/i[@name="e_0_energy"]'
    _forces_path = './/calculation/varray[@name="forces"]'
    _hessian_path = './/calculation/dynmat/varray[@name="hessian"]/v'
    _hessian_unit_path = './/calculation/dynmat/i[@name="unit"]'
    _atomlist_path = './/atominfo/array[@name="atoms"]/set/rc'
    _massdict_path = './/atominfo/array[@name="atomtypes"]/set/rc'

    def read(self, prefix):
        tree = ET.parse(prefix + "/vasprun.xml")
        root = tree.getroot()

        ibrion = root.findall(self._ibrion_path)[0].text.strip()
        cell = np.array([list(map(float, v.text.strip().split())) for v in root.findall(self._cell_path)])

        self.symbols = np.array([], dtype=str)
        for rc in root.findall(self._atomlist_path):
            self.symbols = np.append(self.symbols, rc.findall("c")[0].text.strip())

        self.mass = np.array([], dtype=float).reshape(-1, 1)
        mass_dict = {}
        for rc in root.findall(self._massdict_path):
            c_elements = rc.findall("c")
            mass_dict[c_elements[1].text.strip()] = float(c_elements[2].text.strip())
        for atom in self.symbols:
            self.mass = np.vstack((self.mass, [mass_dict[atom]]))

        if ibrion == "5":
            # calculating hessian. coord, energy and forces should be the first ones
            self.coord = np.array([list(map(float, v.text.strip().split())) for v in root.findall(self._coord_path_in)])
            self.coord = self.coord.dot(cell)  # convert to cartesian coordinates

            self.energy = float(root.findall(self._energy_path)[0].text.strip())

            forces_varray = root.findall(self._forces_path)[0]
            self.grad = -np.array(
                [list(map(float, v.text.strip().split())) for v in forces_varray.findall("v")]
            )  # convert forces to gradients

            self.hess = np.array([list(map(float, v.text.strip().split())) for v in root.findall(self._hessian_path)])
            self.hess = (self.hess + self.hess.T) / 2  # symmetrize hessian
            self.hess = -self.hess  # vasp feature

            if root.findall(self._hessian_unit_path):  # check vasp version
                select_v_list = root.findall(self._select_path)
                if select_v_list:  # check whether some atoms are fixed
                    mass_hess = np.array([], dtype=float)
                    for i, v in enumerate(select_v_list):
                        if "T" in v.text:
                            mass_hess = np.append(mass_hess, self.mass[i, 0])
                    sqrt_mass = np.sqrt(np.repeat(mass_hess, 3))
                else:
                    sqrt_mass = np.sqrt(np.repeat(self.mass, 3))
                mass_divisor = np.outer(sqrt_mass, sqrt_mass)  # mass divisor matrix
                convert_factor = Energy(1, "eV").get("J") / (
                    Mass(1, "amu").get("kg") * np.square(Length(1, "A").get("m")) * (1e24 * 4 * np.square(np.pi))
                )  # conversion factor from THz^2 to eV/(Å^2·amu)
                self.hess = self.hess * mass_divisor / convert_factor  # convert to unweighted hessian
        else:
            # coord, energy and forces should be the last ones
            self.coord = np.array(
                [list(map(float, v.text.strip().split())) for v in root.findall(self._coord_path_out)]
            )
            self.coord = self.coord.dot(cell)  # convert to cartesian coordinates

            self.energy = float(root.findall(self._energy_path)[-1].text.strip())

            forces_varray = root.findall(self._forces_path)[-1]
            self.grad = -np.array(
                [list(map(float, v.text.strip().split())) for v in forces_varray.findall("v")]
            )  # convert forces to gradients
