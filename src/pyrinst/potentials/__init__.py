import contextlib

from .base import POTENTIAL_REGISTRY, Potential, Task
from .fixatom import FixAtom
from .gaussian import Gaussian
from .orca import Orca
from .vasp import Vasp

with contextlib.suppress(ImportError):
    from .mace import MACE

BUILTIN_POTENTIALS = tuple(POTENTIAL_REGISTRY.keys())

__all__ = [
    "FixAtom",
    "Gaussian",
    "Orca",
    "Vasp",
    "POTENTIAL_REGISTRY",
    "Potential",
    "Task",
    "MACE",
    "BUILTIN_POTENTIALS",
]
