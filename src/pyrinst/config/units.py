"""
A module containing unit conversions.

This module provides classes for various unit systems and quantity types
for physical calculations.

To convert a value from a specific unit system to SI:
>>> from polylib import units
>>> au = units.atomic()
>>> time_in_au = 2.0
>>> time_in_seconds = time_in_au * au.time
>>> print(f"{time_in_au} a.u. is equal to {time_in_seconds:.2e} seconds")

To perform direct unit conversions for quantities:
>>> energy_in_hartree = units.Energy(1.0, 'hartree')
>>> energy_in_kcal = energy_in_hartree.get('kcal/mol')
>>> print(f"1.0 Hartree is equal to {energy_in_kcal:.2f} kcal/mol")
"""

import abc
import logging
import math
from scipy import constants as sc

# Import the data from its separate, dedicated module
from ._unit_data import UNIT_REGISTRY

# Set up a logger for this module
log = logging.getLogger(__name__)


class UnitSystem(abc.ABC):
    """
    Abstract Base Class for all unit systems.

    It defines the contract that all concrete unit systems must follow,
    ensuring they provide conversion factors for core physical quantities to SI.
    """
    @property
    @abc.abstractmethod
    def energy(self) -> float:
        """Conversion factor for energy (to Joules)."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def length(self) -> float:
        """Conversion factor for length (to meters)."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def mass(self) -> float:
        """Conversion factor for mass (to kilograms)."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def time(self) -> float:
        """Conversion factor for time (to seconds)."""
        raise NotImplementedError

    @property
    def charge(self) -> float:
        """Conversion factor for charge (to Coulombs). Default is SI."""
        return 1.0

    @property
    def hbar(self) -> float:
        """Value of reduced Planck constant in this unit system."""
        return sc.hbar / (self.energy * self.time)

    @property
    def kb(self) -> float:
        """Value of Boltzmann constant in this unit system."""
        return sc.k / self.energy

    @property
    def e(self) -> float:
        """Value of elementary charge in this unit system."""
        # This is a simplified version; exact conversion depends on permittivity.
        # For most purposes where e is a fundamental unit, it's 1.
        return sc.e / math.sqrt(self.energy * 4 * sc.pi * sc.epsilon_0 * self.length)

    @property
    def amu(self) -> float:
        """Atomic mass unit in this system's mass units."""
        return sc.m_u / self.mass

    def betaTemp(self, beta_or_temp: float) -> float:
        """Converts beta to Temperature or Temperature to beta."""
        return self.energy / (sc.k * beta_or_temp)

    def __eq__(self, other: object) -> bool:
        """Two unit systems are considered equal if they are of the same class."""
        if not isinstance(other, UnitSystem):
            return NotImplemented
        return self.__class__ is other.__class__

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} Unit System>"
    
    @property
    @abc.abstractmethod
    def base_units(self) -> dict[str, str]:
        """
        A dictionary defining the fundamental base units for this system.
        e.g., {'energy': 'eV', 'length': 'A', 'mass': 'amu'}
        """
        raise NotImplementedError

    def Energy(self, value: float) -> 'Energy':
        """Creates an Energy quantity using this system's default energy unit."""
        unit_name = self.base_units['energy']
        return Energy(value, unit_name)

    def Length(self, value: float) -> 'Length':
        unit_name = self.base_units['length']
        return Length(value, unit_name)

    def Mass(self, value: float) -> 'Mass':
        unit_name = self.base_units['mass']
        return Mass(value, unit_name)

    def Time(self, value: float) -> 'Time':
        unit_name = self.base_units['time']
        return Time(value, unit_name)


class SI(UnitSystem):
    """International System of Units (SI)."""
    energy = 1.0  # J
    length = 1.0  # m
    mass = 1.0    # kg
    time = 1.0    # s
    charge = 1.0  # C

    @property
    def hbar(self) -> float:
        return sc.hbar

    @property
    def e(self) -> float:
        return sc.e
    
    @property
    def base_units(self) -> dict[str, str]:
        return {
            'energy': 'J', 
            'length': 'm', 
            'mass': 'kg', 
            'time': 's'
            }


class AtomicUnits(UnitSystem):
    """
    Atomic units (Hartree, Bohr, etc.).
    Fundamental units: mass=m_e, charge=e, ang. momentum=hbar.
    """
    energy = UNIT_REGISTRY['energy']['hartree']
    length = UNIT_REGISTRY['length']['bohr']
    time = UNIT_REGISTRY['time']['au']
    mass = UNIT_REGISTRY['mass']['au']

    @property
    def hbar(self) -> float:
        return 1.0

    @property
    def e(self) -> float:
        return 1.0

    @property
    def base_units(self) -> dict[str, str]:
        return {
            'energy': 'hartree', 
            'length': 'bohr', 
            'mass': 'au', 
            'time': 'au'
            }

class HartreeAngstrom(UnitSystem):
    """
    Atomic units for energy (Hartree) with Angstroms for length.
    Note: In this system, mass is a derived unit.
    """
    energy = UNIT_REGISTRY['energy']['hartree']
    length = UNIT_REGISTRY['length']['angstrom']
    time = UNIT_REGISTRY['time']['au']

    @property
    def mass(self) -> float:
        return self.energy * (self.time**2) / (self.length**2)

    @property
    def base_units(self) -> dict[str, str]:
        return {
            'energy': 'hartree', 
            'length': 'A', 
            'time': 'au'
            }


class KcalAfs(UnitSystem):
    """Energy in kcal/mol, length in Angstrom, time in femtoseconds."""
    energy = UNIT_REGISTRY['energy']['kcal/mol']
    length = UNIT_REGISTRY['length']['angstrom']
    time = UNIT_REGISTRY['time']['fs']

    @property
    def mass(self) -> float:
        return self.energy * (self.time**2) / (self.length**2)

    @property
    def base_units(self) -> dict[str, str]:
        return {
            'energy': 'kcal/mol', 
            'length': 'A', 
            'time': 'fs'
            }


class KcalAamu(UnitSystem):
    """Energy in kcal/mol, length in Angstrom, mass in amu."""
    energy = UNIT_REGISTRY['energy']['kcal/mol']
    length = UNIT_REGISTRY['length']['angstrom']
    mass = UNIT_REGISTRY['mass']['amu']

    @property
    def time(self) -> float:
        return self.length * math.sqrt(self.mass / self.energy)

    @property
    def base_units(self) -> dict[str, str]:
        return {
            'energy': 'kcal/mol', 
            'length': 'A', 
            'mass': 'amu'
            }


class EVAamu(UnitSystem):
    """Energy in eV, length in Angstrom, mass in amu."""
    energy = UNIT_REGISTRY['energy']['eV']
    length = UNIT_REGISTRY['length']['angstrom']
    mass = UNIT_REGISTRY['mass']['amu']

    @property
    def time(self) -> float:
        return self.length * math.sqrt(self.mass / self.energy)

    @property
    def base_units(self) -> dict[str, str]:
        # Time is derived in this system.
        return {
            'energy': 'eV', 
            'length': 'A', 
            'mass': 'amu'
            }


class CmBohrAmu(UnitSystem):
    """Energy in cm^-1, length in bohr, mass in amu."""
    energy = UNIT_REGISTRY['energy']['cm-1']
    length = UNIT_REGISTRY['length']['bohr']
    mass = UNIT_REGISTRY['mass']['amu']
    
    @property
    def time(self) -> float:
        return self.length * math.sqrt(self.mass / self.energy)

    @property
    def base_units(self) -> dict[str, str]:
        return {
            'energy': 'cm-1', 
            'length': 'bohr', 
            'mass': 'amu'
            }


class Quantity:
    """
    A base class for a physical quantity, combining a value and a unit.
    It uses the central UNIT_REGISTRY for all conversion logic.
    """
    def __init__(self, value: float, unit: str, quantity_type: str):
        self._registry = UNIT_REGISTRY.get(quantity_type)
        if unit.lower() not in self._registry:
            raise ValueError(
                f"Unit '{unit}' is not a recognized unit for the physical "
                f"quantity '{quantity_type}'."
            )
        self.value = value
        self.unit = unit
        self.quantity_type = quantity_type

    def get(self, to_unit: str) -> float:
        """Return the value of the quantity in the specified unit."""
        if to_unit.lower() not in self._registry:
            raise ValueError(
                f"Cannot convert to '{to_unit}'; it is not a recognized unit "
                f"for '{self.quantity_type}'."
            )

        from_si_factor = self._registry[self.unit.lower()]
        to_si_factor = self._registry[to_unit.lower()]

        si_value = self.value * from_si_factor
        final_value = si_value / to_si_factor
        return final_value

    def change(self, new_unit: str) -> "Quantity":
        """Return a new Quantity object in the new unit."""
        new_value = self.get(new_unit)
        return self.__class__(new_value, new_unit)

    def __str__(self) -> str:
        return f"{self.value:g} {self.unit}"


class Length(Quantity):
    """Represents a length quantity."""
    def __init__(self, value: float, unit: str = 'm'):
        super().__init__(value, unit, 'length')


class Energy(Quantity):
    """Represents an energy quantity."""
    def __init__(self, value: float, unit: str = 'J'):
        super().__init__(value, unit, 'energy')


class Mass(Quantity):
    """Represents a mass quantity."""
    def __init__(self, value: float, unit: str = 'kg'):
        super().__init__(value, unit, 'mass')


class Time(Quantity):
    """Represents a time quantity."""
    def __init__(self, value: float, unit: str = 's'):
        super().__init__(value, unit, 'time')

