from .base import POTENTIAL_REGISTRY, Potential, Task
from .fixatom import FixAtom
from .gaussian import Gaussian
from .orca import Orca
from .vasp import Vasp

_MACE_IMPORT_ERROR: ImportError | None = None

try:
    from .mace import MACE
except ImportError as exc:
    _MACE_IMPORT_ERROR = exc

BUILTIN_POTENTIALS = tuple(POTENTIAL_REGISTRY.keys())


def __getattr__(name: str):
    if name == "MACE" and _MACE_IMPORT_ERROR is not None:
        msg = (
            "MACE backend is unavailable because its optional dependencies could not be imported. "
            "Install the MACE/ASE stack first."
        )
        raise ImportError(msg) from _MACE_IMPORT_ERROR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "FixAtom",
    "Gaussian",
    "Orca",
    "Vasp",
    "POTENTIAL_REGISTRY",
    "Potential",
    "Task",
    "BUILTIN_POTENTIALS",
]

if _MACE_IMPORT_ERROR is None:
    __all__.append("MACE")
