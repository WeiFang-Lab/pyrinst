import contextlib

from .base import POTENTIAL_REGISTRY, Potential, Task
from .gaussian import Gaussian

with contextlib.suppress(ImportError):
    from .mace import MACE

BUILTIN_POTENTIALS = tuple(POTENTIAL_REGISTRY.keys())

__all__ = ["Gaussian", "POTENTIAL_REGISTRY", "Potential", "Task", "MACE", "BUILTIN_POTENTIALS"]
