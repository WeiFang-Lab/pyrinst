from types import MappingProxyType
from typing import Final
from .abc import OnTheFlyDriver, OnTheFlyResult
from .gaussian import Gaussian

_REGISTRY: dict[str, type[OnTheFlyDriver]] = {
    'gaussian': Gaussian,
}

DRIVER_REGISTRY: Final = MappingProxyType(_REGISTRY)
