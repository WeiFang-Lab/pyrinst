# tests/test_units.py

import pytest
from scipy import constants as sc

from pyrinst.config import units, _unit_data

# For floating point comparisons, it's good to have a tolerance
# pytest.approx is used for this purpose.


# =============================================================================
#  1. TESTS FOR UNIT SYSTEM CLASSES (e.g., AtomicUnits, SI)
# =============================================================================

class TestUnitSystems:
    """Groups tests for the AbstractUnitSystem subclasses."""

    def test_si_system(self):
        """Tests the values of the SI unit system."""
        si = units.SI()
        assert si.energy == 1.0
        assert si.length == 1.0
        assert si.mass == 1.0
        assert si.time == 1.0
        assert si.hbar == sc.hbar
        assert si.e == sc.e

    def test_atomic_units_system(self):
        """Tests the values of the AtomicUnits system against the registry."""
        au = units.AtomicUnits()
        registry = _unit_data.UNIT_REGISTRY

        assert au.energy == pytest.approx(registry['energy']['hartree'])
        assert au.length == pytest.approx(registry['length']['bohr'])
        assert au.mass == pytest.approx(registry['mass']['au'])
        assert au.time == pytest.approx(registry['time']['au'])
        assert au.hbar == 1.0  # In atomic units, hbar is 1

    def test_hartree_angstrom_system(self):
        """Tests a derived system like HartreeAngstrom."""
        ha = units.HartreeAngstrom()
        registry = _unit_data.UNIT_REGISTRY

        # Test that the inherited property is correct
        assert ha.energy == pytest.approx(registry['energy']['hartree'])
        # Test that the overridden property is correct
        assert ha.length == pytest.approx(registry['length']['angstrom'])

    def test_system_equality(self):
        """Tests the __eq__ method of unit systems."""
        au1 = units.AtomicUnits()
        au2 = units.AtomicUnits()
        ha = units.HartreeAngstrom()
        si = units.SI()

        assert au1 == au2
        assert au1 != ha
        assert au1 != si
        assert au1 != "a string"  # Should handle comparison with other types


# =============================================================================
#  2. TESTS FOR QUANTITY CLASSES (e.g., Energy, Length)
# =============================================================================

class TestQuantities:
    """Groups tests for the Quantity subclasses and their conversions."""

    # This parametrized test now covers various casings implicitly
    @pytest.mark.parametrize(
        "value, from_unit, to_unit, expected",
        [
            (1.0, 'hartree', 'eV', sc.value("Hartree energy in eV")),
            (1.0, 'HARTREE', 'ev', sc.value("Hartree energy in eV")), # Test mixed case
            (1.0, 'eV', 'J', sc.e),
            (1.0, 'hartree', 'kcal/mol', 627.509),
            (627.509, 'kcal/mol', 'hartree', 1.0),
            (1.0, 'EV', 'j', sc.e), # Test mixed case
            (10.0, 'eV', 'eV', 10.0), # Identity conversion
        ]
    )
    def test_energy_conversions(self, value, from_unit, to_unit, expected):
        """Tests various energy unit conversions."""
        energy = units.Energy(value, from_unit)
        result = energy.get(to_unit)
        assert result == pytest.approx(expected, rel=1e-4) # Use relative tolerance

    @pytest.mark.parametrize(
        "value, from_unit, to_unit, expected",
        [
            (1.0, 'bohr', 'A', sc.value("Bohr radius") / sc.angstrom),
            (1.0, 'A', 'm', sc.angstrom),
            (1.0, 'bohr', 'm', sc.value("Bohr radius")),
            (5.0, 'A', 'A', 5.0), # Identity conversion
        ]
    )
    def test_length_conversions(self, value, from_unit, to_unit, expected):
        """Tests various length unit conversions."""
        length = units.Length(value, from_unit)
        result = length.get(to_unit)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_invalid_unit_instantiation(self):
        """Tests that creating a Quantity with an unknown unit raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            units.Energy(1.0, "Weilometer")
        # Check if the error message is informative
        assert "Weilometer" in str(excinfo.value)
        assert "energy" in str(excinfo.value)

    def test_invalid_unit_conversion(self):
        """Tests that converting to an unknown unit raises ValueError."""
        energy = units.Energy(1.0, 'eV')
        with pytest.raises(ValueError) as excinfo:
            energy.get("FluxCapacitorUnits")
        assert "FluxCapacitorUnits" in str(excinfo.value)


# =============================================================================
#  3. TESTS FOR BACKWARD COMPATIBILITY
# =============================================================================

def test_backward_compatibility_aliases():
    """
    Ensures that old, non-PEP8 names still point to the correct classes
    and function as expected. This is crucial for not breaking existing code.
    """
    # Check that the alias points to the correct class
    assert units.atomic is units.AtomicUnits
    assert units.hartAng is units.HartreeAngstrom
    assert units.eVAamu is units.EVAamu

    # Check that instantiating via an alias works correctly
    ha_via_alias = units.hartAng()
    ha_via_new_name = units.HartreeAngstrom()

    assert ha_via_alias.energy == ha_via_new_name.energy
    assert ha_via_alias.length == ha_via_new_name.length
    assert ha_via_alias == ha_via_new_name