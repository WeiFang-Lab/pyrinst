import contextlib

from .base import POTENTIAL_REGISTRY, Potential, Task
from .gaussian import Gaussian
from .orca import Orca

with contextlib.suppress(ImportError):
    from .mace import MACE

BUILTIN_POTENTIALS = tuple(POTENTIAL_REGISTRY.keys())

__all__ = ["Gaussian", "Orca", "POTENTIAL_REGISTRY", "Potential", "Task", "MACE", "BUILTIN_POTENTIALS"]
