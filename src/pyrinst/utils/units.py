"""
A toolkit for physical unit conversions.

This module provides the `Quantity` class and its subclasses (`Energy`,
`Length`, `Mass`, `Time`). These classes are used to encapsulate a
numerical value with its unit, and to convert between different units.

This module *only* provides conversion tools. It does not define the
internal unit standard for the `pyrinst` package.

The project's internal standard (Atomic Units) and the conversion factors
used at the I/O boundaries are defined in `pyrinst.config.constants`.

Examples
--------
To perform a one-off conversion:

>>> from pyrinst.config.units import Energy, Length
>>>
>>> # Convert 1.0 kcal/mol to Hartree
>>> kcal_to_hartree = Energy(1.0, 'kcal/mol').get('Hartree')
>>>
>>> # Create a 2.5 Angstrom Length object
>>> length_obj = Length(2.5, 'A')
>>> # Convert it to a new object in Bohr
>>> length_in_bohr = length_obj.change('Bohr')
"""

import logging
from typing import Self

from pyrinst.config._unit_data import UNIT_REGISTRY

log = logging.getLogger(__name__)


class Quantity:
    """
    Base class for a physical quantity, encapsulating a value and a unit.

    This class provides a robust, object-oriented way to handle physical
    quantities and serves as the foundation for unit conversions. All
    conversion logic is powered by the `UNIT_REGISTRY`.

    Attributes
    ----------
    value : float
        The numerical value of the quantity.
    unit : str
        The unit associated with the value (e.g., 'eV', 'A').
    quantity_type : str
        The type of physical quantity (e.g., 'energy', 'length').
    """

    def __init__(self, value: float, unit: str, quantity_type: str):
        """
        Initializes the Quantity object.

        Parameters
        ----------
        value : float
            The numerical value of the quantity.
        unit : str
            The unit associated with the value.
        quantity_type : str
            The type of physical quantity (must be a key in UNIT_REGISTRY).

        Raises
        ------
        ValueError
            If the `unit` is not a recognized unit for the `quantity_type`.
        """
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
        """
        Return the numerical value of the quantity in a specified unit.

        This is the general-purpose method for converting between any two
        known units. It returns a raw float for direct use in calculations.

        Parameters
        ----------
        to_unit : str
            The string name of the target unit (e.g., 'eV', 'cm-1').

        Returns
        -------
        float
            The numerical value of the quantity in the target unit.

        Raises
        ------
        ValueError
            If `to_unit` is not a recognized unit.

        Examples
        --------
        >>> e = Energy(1.0, 'Hartree')
        >>> e.get('kcal/mol')
        627.509...
        """
        if to_unit.lower() not in self._registry:
            raise ValueError(
                f"Cannot convert to '{to_unit}'; it is not a recognized unit "
                f"for '{self.quantity_type}'."
            )

        # Convert value to SI, then from SI to target unit
        from_si_factor = self._registry[self.unit]
        to_si_factor = self._registry[to_unit]
        si_value = self.value * from_si_factor
        return si_value / to_si_factor

    def change(self, new_unit: str) -> Self:
        """
        Create a new Quantity object converted to a new unit.

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
            unit and value.

        Examples
        --------
        >>> l_angstrom = Length(2.0, 'A')
        >>> l_bohr = l_angstrom.change('Bohr')
        >>> print(l_bohr)
        3.779... Bohr
        """
        new_value = self.get(new_unit)
        # Create a new instance of the *current* class (e.g., Length)
        return self.__class__(new_value, new_unit)

    def __str__(self) -> str:
        """Return a user-friendly string representation."""
        return f"{self.value:g} {self.unit}"


class Length(Quantity):
    """
    A `Quantity` subclass representing length.
    """
    def __init__(self, value: float, unit: str = 'm'):
        """
        Initializes the Length object.

        Parameters
        ----------
        value : float
            The numerical value of the length.
        unit : str, optional
            The unit of length, by default 'm' (meters).
        """
        super().__init__(value, unit, 'length')


class Energy(Quantity):
    """
    A `Quantity` subclass representing energy.
    """
    def __init__(self, value: float, unit: str = 'J'):
        """
        Initializes the Energy object.

        Parameters
        ----------
        value : float
            The numerical value of the energy.
        unit : str, optional
            The unit of energy, by default 'J' (Joules).
        """
        super().__init__(value, unit, 'energy')


class Mass(Quantity):
    """
    A `Quantity` subclass representing mass.
    """
    def __init__(self, value: float, unit: str = 'kg'):
        """
        Initializes the Mass object.

        Parameters
        ----------
        value : float
            The numerical value of the mass.
        unit : str, optional
            The unit of mass, by default 'kg' (kilograms).
        """
        super().__init__(value, unit, 'mass')


class Time(Quantity):
    """
    A `Quantity` subclass representing time.
    """
    def __init__(self, value: float, unit: str = 's'):
        """
        Initializes the Time object.

        Parameters
        ----------
        value : float
            The numerical value of the time.
        unit : str, optional
            The unit of time, by default 's' (seconds).
        """
        super().__init__(value, unit, 'time')
