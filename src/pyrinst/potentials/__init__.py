from .base import POTENTIAL_REGISTRY, Level, Potential
from .executors import CachedExecutor, Driver, Executor, FixAtom, ParallelExecutor, SerialExecutor, SingleExecutor
from .gaussian import Gaussian
from .orca import Orca
from .vasp import Vasp

_MACE_IMPORT_ERROR: ImportError | None = None

try:
    from .mace import MACE as _MACE
except ImportError as exc:
    _MACE_IMPORT_ERROR = exc
else:
    MACE = _MACE

BUILTIN_POTENTIALS = tuple(POTENTIAL_REGISTRY.keys())


def __getattr__(name: str):
    if name == "MACE" and _MACE_IMPORT_ERROR is not None:
        msg = (
            "MACE backend is unavailable because its optional dependencies could not be imported. "
            "Install the project-provided MACE wheel and its matching runtime environment first."
        )
        raise ImportError(msg) from _MACE_IMPORT_ERROR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "FixAtom",
    "Gaussian",
    "Orca",
    "Driver",
    "Vasp",
    "POTENTIAL_REGISTRY",
    "Executor",
    "SingleExecutor",
    "SerialExecutor",
    "ParallelExecutor",
    "CachedExecutor",
    "Potential",
    "Level",
    "BUILTIN_POTENTIALS",
]

if _MACE_IMPORT_ERROR is None:
    __all__.append("MACE")
