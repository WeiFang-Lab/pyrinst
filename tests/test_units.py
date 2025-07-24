"""
Test suite for the refactored units module.

This file contains a series of tests to ensure the correctness and robustness
of the unit systems, quantity classes, and factory methods defined in the
units module.
"""

import math
import pytest
from scipy import constants as sc

from pyrinst.config import units, _unit_data


# TESTS FOR UNIT SYSTEM CLASSES (e.g., AtomicUnits, EVAamu)

class TestUnitSystems:
    """
    Groups tests for the UnitSystem subclasses.
    Verifies the correctness of their conversion factors and properties.
    """

    def test_si_system_values(self):
        """Tests the values of the SI unit system."""
        si = units.SI()
        assert si.energy == 1.0
        assert si.length == 1.0
        assert si.mass == 1.0
        assert si.time == 1.0
        assert si.hbar == sc.hbar
        assert si.e == sc.e
        assert si.base_units == {'energy': 'J', 'length': 'm', 'mass': 'kg', 'time': 's'}

    def test_atomic_units_system_values(self):
        """Tests the values of the AtomicUnits system."""
        au = units.AtomicUnits()
        registry = _unit_data.UNIT_REGISTRY
        assert au.energy == pytest.approx(registry['energy']['hartree'])
        assert au.length == pytest.approx(registry['length']['bohr'])
        assert au.mass == pytest.approx(registry['mass']['au'])
        assert au.time == pytest.approx(registry['time']['au'])
        assert au.hbar == 1.0

    def test_kcalafs_derived_mass(self):
        """Ensures the derived mass in KcalAfs is physically consistent."""
        system = units.KcalAfs()
        expected_mass = system.energy * (system.time ** 2) / (system.length ** 2)
        assert system.mass == pytest.approx(expected_mass)

    def test_kcalaamu_derived_time(self):
        """Ensures the derived time in KcalAamu is physically consistent."""
        system = units.KcalAamu()
        expected_time = system.length * math.sqrt(system.mass / system.energy)
        assert system.time == pytest.approx(expected_time)

    def test_cmbohramu_derived_time(self):
        """Ensures the derived time in CmBohrAmu is physically consistent."""
        system = units.CmBohrAmu()
        expected_time = system.length * math.sqrt(system.mass / system.energy)
        assert system.time == pytest.approx(expected_time)

    def test_system_equality(self):
        """Tests the equality check, which should be based on class type."""
        au1 = units.AtomicUnits()
        au2 = units.AtomicUnits()
        ha = units.HartreeAngstrom()

        assert au1 == au2
        assert au1 != ha
        assert au1 != "some string"  # Should handle comparison with other types

    def test_base_units_property(self):
        """Tests that the base_units property is correctly defined."""
        eva = units.EVAamu()
        ha = units.HartreeAngstrom()

        assert eva.base_units == {'energy': 'eV', 'length': 'A', 'mass': 'amu'}
        # Mass and Time are derived, so they should not be in base_units
        assert 'time' not in eva.base_units

        # Mass is derived in HartreeAngstrom, so it should not be a base unit
        assert 'mass' not in ha.base_units


# TESTS FOR FACTORY METHODS (e.g., system.Energy())

class TestFactoryMethods:
    """Tests the factory methods (.Energy(), .Length(), etc.) on UnitSystem."""

    def test_factory_method_on_base_unit(self):
        """
        Tests that factory methods work correctly when called for a
        fundamental unit of a system.
        """
        eva_system = units.EVAamu()
        # Create an energy object using the factory; no unit string needed.
        energy = eva_system.Energy(5.0)

        assert isinstance(energy, units.Energy)
        assert energy.unit == 'eV'
        assert energy.value == 5.0
        # Check conversion to verify the object is functional
        assert energy.get('J') == pytest.approx(5.0 * sc.e)

    def test_factory_method_failure_on_derived_unit(self):
        """
        Tests that calling a factory method for a derived unit (which has no
        simple name) correctly raises a KeyError. This is expected behavior.
        """
        ha_system = units.HartreeAngstrom()
        # The 'mass' unit in HartreeAngstrom is derived and has no base name.
        # Therefore, calling .Mass() should fail, which is correct.
        with pytest.raises(KeyError) as excinfo:
            ha_system.Mass(1.0)
        # Check that the error message clearly indicates the missing key.
        assert "'mass'" in str(excinfo.value)


# TESTS FOR QUANTITY CLASSES (e.g., Energy, Length)

class TestQuantities:
    """
    Tests the Quantity subclasses for direct unit conversions and error handling
    with a wide range of examples.
    """

    @pytest.mark.parametrize(
        "value, from_unit, to_unit, expected",
        [
            # --- Energy Conversions ---
            (1.0, 'hartree', 'eV', sc.value("Hartree energy in eV")),
            (27.21138, 'eV', 'hartree', 1.0),
            (1.0, 'hartree', 'kcal/mol', 627.509),
            (1.0, 'kcal/mol', 'J', sc.calorie * 1000 / sc.N_A),
            (1.0, 'eV', 'J', sc.e),
            (1.0, 'J', 'erg', 1e7),

            # --- Conversions to/from Wavenumbers (cm-1) ---
            (1.0, 'hartree', 'cm-1', 219474.63),
            (1.0, 'eV', 'cm-1', 8065.54),
            (1.0, 'kcal/mol', 'cm-1', 349.755),
            (1.0, 'J', 'cm-1', 1 / (sc.h * sc.c * 100)),
            (1000.0, 'cm-1', 'eV', 0.123984),

            # --- Length Conversions ---
            (1.0, 'bohr', 'A', sc.value("Bohr radius") / sc.angstrom),
            (1.0, 'A', 'm', sc.angstrom),
            (1.0, 'in', 'cm', 2.54),
            (1.0, 'm', 'bohr', 1 / sc.value("Bohr radius")),

            # --- Mass and Time Conversions ---
            (1.0, 'amu', 'kg', sc.m_u),
            (1.0, 'Da', 'g', sc.m_u * 1000),
            (1.0, 'fs', 's', sc.femto),
            (1.0, 's', 'ps', 1e12),

            # --- Identity and Case-Insensitivity ---
            (10.0, 'eV', 'eV', 10.0),
            (1.0, 'HARTREE', 'ev', sc.value("Hartree energy in eV")),
        ]
    )
    def test_extensive_conversions(self, value, from_unit, to_unit, expected):
        """Tests a wide range of unit conversions for correctness."""
        # Determine quantity type based on unit
        registry = _unit_data.UNIT_REGISTRY
        if from_unit.lower() in registry['energy']:
            quantity = units.Energy(value, from_unit)
        elif from_unit.lower() in registry['length']:
            quantity = units.Length(value, from_unit)
        elif from_unit.lower() in registry['mass']:
            quantity = units.Mass(value, from_unit)
        else:
            quantity = units.Time(value, from_unit)

        result = quantity.get(to_unit)
        assert result == pytest.approx(expected, rel=1e-5)

    def test_invalid_unit_instantiation(self):
        """
        Tests that creating a Quantity with an unknown unit raises ValueError.
        """
        with pytest.raises(ValueError) as excinfo:
            units.Energy(1.0, "invalid_unit")
        assert "invalid_unit" in str(excinfo.value)

    def test_invalid_unit_conversion(self):
        """
        Tests that converting to an unknown unit raises ValueError.
        """
        length = units.Length(1.0, 'm')
        with pytest.raises(KeyError) as excinfo:
            length.get("invalid_length_unit")
        assert "invalid_length_unit" in str(excinfo.value)


# TESTS FOR THE UnitSystem.convert() METHOD

class TestSystemConvertMethod:
    """
    Tests the functionality of converting a Quantity object from one
    unit system's context to another using the `UnitSystem.convert()` method.
    """

    def test_convert_length_between_systems(self):
        """
        Tests converting a Length object from AtomicUnits (bohr) to
        the HartreeAngstrom system (which uses Angstrom).
        """
        # 1. Define source and target unit systems
        au_system = units.AtomicUnits()
        ha_system = units.HartreeAngstrom()

        # 2. Create a length object from the source system
        length_in_au = au_system.Length(2.0)  # This is a 2.0 bohr Quantity

        # 3. Use the target system's .convert() method
        converted_length_obj = ha_system.convert(length_in_au)

        # 4. Assert the returned object is of the correct type and has the correct unit
        assert isinstance(converted_length_obj, units.Length)
        # The target unit should be 'A' as defined in HartreeAngstrom.base_units
        assert converted_length_obj.unit.lower() == 'a'

        # 5. Assert the value is correct by manual calculation
        expected_value = (2.0 * sc.value("Bohr radius")) / sc.angstrom
        assert converted_length_obj.value == pytest.approx(expected_value)

    def test_convert_energy_and_verify_functionality(self):
        """
        Tests converting an Energy object and ensures the resulting object
        is fully functional by calling .get() on it.
        """
        eva_system = units.EVAamu()
        au_system = units.AtomicUnits()

        # Create a 1.0 eV energy object
        energy_in_eva = eva_system.Energy(1.0)

        # Convert it to the AtomicUnits system
        energy_in_au = au_system.convert(energy_in_eva)

        # Assert basic properties of the new object
        assert isinstance(energy_in_au, units.Energy)
        assert energy_in_au.unit == 'hartree'
        assert energy_in_au.value == pytest.approx(1.0 / sc.value("Hartree energy in eV"))

        # CRITICAL: Verify the new object works as expected
        # Can it convert its value to yet another unit?
        value_in_kcal = energy_in_au.get('kcal/mol')
        assert value_in_kcal == pytest.approx(1.0 / sc.value("Hartree energy in eV") * 627.509)

    def test_convert_failure_for_derived_unit_target(self):
        """
        Tests that .convert() correctly raises a KeyError when the target
        system does not have a defined base unit for the quantity type.
        This is a critical safety feature.
        """
        au_system = units.AtomicUnits()
        ha_system = units.HartreeAngstrom()

        # Create a mass object in a system where mass is a base unit
        mass_in_au = au_system.Mass(1.0)  # This is 1.0 electron mass

        # Attempt to convert it to the HartreeAngstrom system, where
        # mass is a derived unit and has no entry in `base_units`.
        with pytest.raises(KeyError) as excinfo:
            ha_system.convert(mass_in_au)

        # The error should clearly indicate that 'mass' is not a base unit
        # in the target system.
        assert "'mass'" in str(excinfo.value)