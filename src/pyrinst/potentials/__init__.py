import contextlib

from .base import POTENTIAL_REGISTRY, Potential, Task

with contextlib.suppress(ImportError):
    from .mace import MACE

BUILTIN_POTENTIALS = tuple(POTENTIAL_REGISTRY.keys())

__all__ = ["POTENTIAL_REGISTRY", "Potential", "Task", "MACE", "BUILTIN_POTENTIALS"]
