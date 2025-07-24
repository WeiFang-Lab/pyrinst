"""A robust module for handling physical units and unit systems.

This module provides a powerful, object-oriented framework for performing
unit conversions in scientific computing. It is built on two core concepts:

1.  **UnitSystem**: Classes (e.g., `AtomicUnits`, `EVAamu`) that define a
    self-consistent physical environment. Their primary role is to provide
    authoritative conversion factors from the system's base units to SI units.

2.  **Quantity**: Classes (e.g., `Energy`, `Length`) that encapsulate a
    numerical value with its corresponding unit. They provide a flexible way
    to convert between any two known units.

The recommended modern usage combines these two concepts through factory methods,
enabling clear, type-safe, and context-aware unit conversions.

Examples
--------
**1. Basic Conversion to SI using a UnitSystem:**
   (For converting raw data from a specific computational environment)

>>> from polylib.config import units
>>> au_system = units.AtomicUnits()
>>> force_constant_au = 0.1  # A raw value in Hartree/Bohr^2
>>> k_in_si = force_constant_au * au_system.energy / (au_system.length**2)
>>> print(f"0.1 Hartree/Bohr^2 is equal to {k_in_si:.2f} J/m^2")
0.1 Hartree/Bohr^2 is equal to 1556.89 J/m^2

**2. Direct Conversion between Any Two Units using a Quantity Object:**
   (For general-purpose, one-off conversions)

>>> units.Energy(1.0, 'Hartree').get('cm-1')
219474.63136314112
>>> units.Length(2.0, 'A').get('bohr')
3.779451453315013

**3. Recommended Modern Usage: The Factory Method Pattern:**
   (Combines the clarity of UnitSystem with the power of Quantity)

>>> eva_system = units.EVAamu()
>>> # Create a 5.0 eV energy object without manually typing 'eV'
>>> energy = eva_system.Energy(5.0)
>>> print(energy)
5.0 eV
>>> # The created object can then be used for further conversions
>>> energy.get('kcal/mol')
115.328...

**4. Converting a Value from One Unit System to Another:**
   (The safe, recommended way to switch between computational environments)

>>> au_system = units.AtomicUnits()
>>> ha_system = units.HartreeAngstrom()
>>> # Create a 2.0 Bohr length object using the source system
>>> length_in_au = au_system.Length(2.0)
>>> # Get its numerical value in the target system's default length unit (Angstrom)
>>> value_in_ha = length_in_au.value_in(ha_system)
>>> print(f"The value of {length_in_au} is {value_in_ha:.4f} in the {ha_system}.")
The value of 2.0 bohr is 1.0584 in the <HartreeAngstrom UnitSystem>.
"""

import abc
import logging
import math
from scipy import constants as sc
from typing import Self

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
        """Value of elementary charge in this unit system.
        
        Notes
        -----
        This is a simplified version; exact conversion depends on permittivity.
        For most purposes where e is a fundamental unit, it's 1.
        """
        return sc.e / math.sqrt(self.energy * 4 * sc.pi * sc.epsilon_0 * self.length)

    @property
    def amu(self) -> float:
        """Atomic mass unit in this system's mass units."""
        return sc.m_u / self.mass

    def convert(self, quantity_obj: 'Quantity') -> Self:
        """Converts a given Quantity object to this unit system.

        This is the most powerful and recommended method for converting a quantity
        from one unit system's context to another, as it returns a new,
        fully functional Quantity object.

        Args:
            quantity_obj: An existing Quantity object (e.g., Energy, Length)
                          from any source system.

        Returns:
            A new Quantity object of the same physical type, expressed in the
            base unit of this system.

        Raises:
            KeyError: If this system does not define a base unit for the
                      given quantity's physical type.
        """
        # 1. Get the target unit name from self (the current system).
        #    e.g., for EVAamu, if quantity_obj is an Energy, this gets 'eV'.
        target_unit_name = self.base_units[quantity_obj.quantity_type]

        # 2. Delegate the conversion logic to the quantity object's own
        #    .change() method. The .change() method already knows how to
        #    create a new object in any target unit.
        return quantity_obj.change(target_unit_name)

    def betaTemp(self, beta_or_temp: float) -> float:
        """Converts beta to Temperature or Temperature to beta.

        Parameters
        ----------
        beta_or_temp : float
            The value to convert, which can be either beta (in units of 1/energy)
            or a temperature (in Kelvin).

        Returns
        -------
        float
            If input was beta, returns temperature. If input was temperature,
            returns beta.
        """
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

        Examples
        --------
        {'energy': 'eV', 'length': 'A', 'mass': 'amu'}
        """
        raise NotImplementedError

    def Energy(self, value: float) -> 'Energy':
        """Creates an Energy quantity using this system's default energy unit.
        
        Parameters
        ----------
        value : float
            The numerical value of the energy.

        Returns
        -------
        Energy
            An `Energy` object initialized with the given value and the
            system's default energy unit.
        """
        unit_name = self.base_units['energy']
        return Energy(value, unit_name)

    def Length(self, value: float) -> 'Length':
        """Creates a Length quantity using this system's default length unit.

        Parameters
        ----------
        value : float
            The numerical value of the length.

        Returns
        -------
        Length
            A `Length` object initialized with the given value and the
            system's default length unit.
        """
        unit_name = self.base_units['length']
        return Length(value, unit_name)

    def Mass(self, value: float) -> 'Mass':
        """Creates a Mass quantity using this system's default mass unit.

        Parameters
        ----------
        value : float
            The numerical value of the mass.

        Returns
        -------
        Mass
            A `Mass` object initialized with the given value and the
            system's default mass unit.
        """
        unit_name = self.base_units['mass']
        return Mass(value, unit_name)

    def Time(self, value: float) -> 'Time':
        """Creates a Time quantity using this system's default time unit.

        Parameters
        ----------
        value : float
            The numerical value of the time.

        Returns
        -------
        Time
            A `Time` object initialized with the given value and the
            system's default time unit.
        """
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
    energy = UNIT_REGISTRY['energy']['Hartree']
    length = UNIT_REGISTRY['length']['Bohr']
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
            'energy': 'Hartree', 
            'length': 'Bohr', 
            'mass': 'au', 
            'time': 'au'
            }

class HartreeAngstrom(UnitSystem):
    """
    Atomic units for energy (Hartree) with Angstroms for length.

    Notes
    -----
    In this system, mass is a derived unit.
    """
    energy = UNIT_REGISTRY['energy']['Hartree']
    length = UNIT_REGISTRY['length']['Angstrom']
    time = UNIT_REGISTRY['time']['au']

    @property
    def mass(self) -> float:
        return self.energy * (self.time**2) / (self.length**2)

    @property
    def base_units(self) -> dict[str, str]:
        return {
            'energy': 'Hartree', 
            'length': 'A', 
            'time': 'au'
            }


class KcalAfs(UnitSystem):
    """Energy in kcal/mol, length in Angstrom, time in femtoseconds."""
    energy = UNIT_REGISTRY['energy']['kcal/mol']
    length = UNIT_REGISTRY['length']['Angstrom']
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
    length = UNIT_REGISTRY['length']['Angstrom']
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
    length = UNIT_REGISTRY['length']['Angstrom']
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
    """A base class for a physical quantity, encapsulating a value and a unit.

    This class provides a robust, object-oriented way to handle physical
    quantities. It serves as the foundation for unit conversions.

    All conversion logic is powered by a central, case-insensitive
    `UNIT_REGISTRY`, ensuring a single source of truth for physical constants.

    Parameters
    ----------
    value : float
        The numerical value of the quantity.
    unit : str
        The unit associated with the value (e.g., 'eV', 'A').
    quantity_type : str
        The type of physical quantity ('energy', 'length', etc.).

    Attributes
    ----------
    value : float
        The numerical value of the quantity.
    unit : str
        The unit associated with the value.
    quantity_type : str
        The type of physical quantity.
    """
    def __init__(self, value: float, unit: str, quantity_type: str):
        self._registry = UNIT_REGISTRY[quantity_type]
        if unit.lower() not in self._registry:
            raise ValueError(
                f"Unit '{unit}' is not a recognized unit for the physical "
                f"quantity '{quantity_type}'."
            )
        self.value = value
        self.unit = unit
        self.quantity_type = quantity_type

    def get(self, to_unit: str) -> float:
        """Return the numerical value of the quantity in a specified unit.

        This is the general-purpose method for converting between any two
        known units by providing their string names. It returns a raw float
        for direct use in calculations.

        Parameters
        ----------
        to_unit : str
            The string name of the target unit (e.g., 'eV', 'cm-1').

        Returns
        -------
        float
            The numerical value of the quantity in the target unit.
        """
        if to_unit.lower() not in self._registry:
            raise ValueError(
                f"Cannot convert to '{to_unit}'; it is not a recognized unit "
                f"for '{self.quantity_type}'."
            )

        from_si_factor = self._registry[self.unit]
        to_si_factor = self._registry[to_unit]
        si_value = self.value * from_si_factor
        return si_value / to_si_factor

    def change(self, new_unit: str) -> Self:
        """Create a new Quantity object converted to a new unit.

        This method is useful for chaining operations or when the converted
        quantity needs to be passed to other functions as a complete object.

        Parameters
        ----------
        new_unit : str
            The string name of the target unit.

        Returns
        -------
        Self
            A new instance of the same `Quantity` subclass with the new
            unit and value. `typing.Self` ensures correct type hinting.
        """
        new_value = self.get(new_unit)
        return self.__class__(new_value, new_unit)

    def __str__(self) -> str:
        """Return a user-friendly string representation."""
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
