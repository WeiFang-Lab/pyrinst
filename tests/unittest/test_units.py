"""
Unit tests for the unit conversion toolkit in `pyrinst.utils.units`.

Design Logic:
1.  This is a pure unit test for `units.py`. It does NOT test `constants.py`
    or any other part of the application.
2.  The `units.py` module's main dependency is the `_unit_data.py`
    database. We import this database as the "source of truth" to
    validate the conversion logic.
3.  Tests are grouped by functionality:
    - `TestQuantityBase`: Checks error handling and type-agnostic
      methods like `change()`.
    - `TestQuantitySubclasses`: Ensures subclasses (`Energy`, `Length`, etc.)
      are set up correctly.
    - `TestQuantityConversions`: Uses parameterization to verify the
      mathematical correctness of a wide range of conversions.
"""

import pytest

# The database we are testing against
from pyrinst.config import _unit_data

# The module we are testing
from pyrinst.utils import units

# --- 1. Load the "source of truth" from the project's database ---
# We will test that the units.py module can correctly use these factors.
REGISTRY = _unit_data.UNIT_REGISTRY

# Get SI conversion factors for cross-checking
HARTREE_SI = REGISTRY['energy']['hartree']
EV_SI = REGISTRY['energy']['ev']
KCAL_MOL_SI = REGISTRY['energy']['kcal/mol']
CM_INV_SI = REGISTRY['energy']['cm-1']

BOHR_SI = REGISTRY['length']['bohr']
ANGSTROM_SI = REGISTRY['length']['angstrom']

AMU_SI = REGISTRY['mass']['amu']
AU_MASS_SI = REGISTRY['mass']['au']  # This is m_e

AU_TIME_SI = REGISTRY['time']['au']
FS_SI = REGISTRY['time']['fs']
PS_SI = REGISTRY['time']['ps']

KB_AU_HARTREE_PER_KELVIN = (
    _unit_data.UNIT_REGISTRY['energy']['K'] /
    _unit_data.UNIT_REGISTRY['energy']['Hartree']
)


class TestQuantityBase:
    """Tests the core, type-agnostic functionality of the Quantity base class."""

    def test_quantity_initialization_invalid_unit(self) -> None:
        """
        Tests that initializing a Quantity with a unit not in the
        database raises a ValueError.
        """
        with pytest.raises(ValueError, match="not a recognized unit"):
            units.Quantity(1.0, "NotARealUnit", "energy")

    def test_get_invalid_target_unit(self) -> None:
        """
        Tests that calling .get() with an invalid target unit
        raises a ValueError.
        """
        # Create a valid object
        e = units.Energy(1.0, "eV")
        # Try to convert to an invalid unit
        with pytest.raises(ValueError, match="Cannot convert to"):
            e.get("NotARealUnit")

    def test_change_method(self) -> None:
        """
        Tests that the .change() method correctly returns a new object
        of the same class with the converted value and new unit.
        """
        e_hartree = units.Energy(1.0, "Hartree")
        e_ev = e_hartree.change("eV")

        # Check that it returned the correct class
        assert isinstance(e_ev, units.Energy)

        # Check that the new object has the correct attributes
        assert e_ev.unit == "eV"
        assert e_ev.value == pytest.approx(e_hartree.get("eV"))

        # Check that the original object is unchanged
        assert e_hartree.unit == "Hartree"
        assert e_hartree.value == 1.0

    def test_str_representation(self) -> None:
        """Tests the __str__ method for a clean output."""
        e = units.Energy(2.5, "eV")
        assert str(e) == "2.5 eV"

        # Test the 'g' formatting for large/small numbers
        assert str(units.Length(1.0e-10, "m")) == "1e-10 m"


class TestQuantitySubclasses:
    """Tests the specific subclasses: Energy, Length, Mass, Time."""

    def test_subclasses_set_correct_quantity_type(self) -> None:
        """
        Tests that each subclass correctly initializes its
        parent with the right `quantity_type` string.
        """
        assert units.Energy(1, 'eV').quantity_type == 'energy'
        assert units.Length(1, 'A').quantity_type == 'length'
        assert units.Mass(1, 'amu').quantity_type == 'mass'
        assert units.Time(1, 'fs').quantity_type == 'time'

    def test_subclasses_use_correct_default_unit(self) -> None:
        """
        Tests that subclasses default to the correct SI unit
        if one is not provided.
        """
        assert units.Energy(1.0).unit == 'J'
        assert units.Length(1.0).unit == 'm'
        assert units.Mass(1.0).unit == 'kg'
        assert units.Time(1.0).unit == 's'


# --- 3. The main conversion logic test ---

@pytest.mark.parametrize(
    "Cls, value, from_unit, to_unit, expected",
    [
        # --- Energy Conversions ---
        (units.Energy, 1.0, 'hartree', 'ev', HARTREE_SI / EV_SI),
        (units.Energy, 1.0, 'ev', 'hartree', EV_SI / HARTREE_SI),
        (units.Energy, 1.0, 'kcal/mol', 'J', KCAL_MOL_SI),
        (units.Energy, 1.0, 'J', 'kcal/mol', 1.0 / KCAL_MOL_SI),
        (units.Energy, 1.0, 'hartree', 'cm-1', HARTREE_SI / CM_INV_SI),
        (units.Energy, 1000.0, 'cm-1', 'ev', (1000.0 * CM_INV_SI) / EV_SI),

        # --- Length Conversions ---
        (units.Length, 1.0, 'bohr', 'A', BOHR_SI / ANGSTROM_SI),
        (units.Length, 1.0, 'A', 'bohr', ANGSTROM_SI / BOHR_SI),
        (units.Length, 1.0, 'm', 'bohr', 1.0 / BOHR_SI),

        # --- Mass Conversions ---
        (units.Mass, 1.0, 'amu', 'au', AMU_SI / AU_MASS_SI), # amu to m_e
        (units.Mass, 1.0, 'au', 'amu', AU_MASS_SI / AMU_SI), # m_e to amu
        (units.Mass, 1.0, 'kg', 'amu', 1.0 / AMU_SI),

        # --- Time Conversions ---
        (units.Time, 1.0, 'fs', 'au', FS_SI / AU_TIME_SI),
        (units.Time, 1.0, 'au', 'fs', AU_TIME_SI / FS_SI),
        (units.Time, 1.0, 'ps', 's', PS_SI),

        # --- Identity and Case-Insensitivity ---
        (units.Energy, 12.34, 'eV', 'eV', 12.34),
        (units.Length, 1.0, 'A', 'angstrom', 1.0),
        (units.Energy, 1.0, 'HARTREE', 'ev', HARTREE_SI / EV_SI), # Mixed class/case
    ]
)
def test_quantity_conversions(
    cls: type,
    value: float,
    from_unit: str,
    to_unit: str,
    expected: float
) -> None:
    """
    Tests a wide range of conversions for mathematical correctness.

    This parameterized test creates a Quantity object of type `cls` and
    verifies that its `.get()` method produces the `expected` value.
    The expected values are calculated from the SI factors loaded from
    `_unit_data.py`, ensuring the test logic is self-consistent
    with the project's database.
    """
    # Create the object, e.g., units.Energy(1.0, 'hartree')
    obj = cls(value, from_unit)

    # Perform the conversion, e.g., obj.get('ev')
    result = obj.get(to_unit)

    # Check against the expected value
    assert result == pytest.approx(expected)

class TestTemperature:
    """
    Unit tests for the static `Temperature` utility class.

    These tests assume the `Temperature` class exists in `pyrinst.utils.units`
    and that it self-manages its own `_kb_au` constant.
    """

    def test_to_beta_happy_path(self) -> None:
        """
        Tests the T (Kelvin) -> beta (1/Hartree) conversion.
        """
        temp_k = 300.0
        # This is the expected calculation: 1.0 / (T * Kb_au)
        expected_beta = 1.0 / (300.0 * KB_AU_HARTREE_PER_KELVIN)

        # We call the static method on the units.Temperature class
        assert units.Temperature.to_beta(temp_k) == pytest.approx(expected_beta)

    def test_to_kelvin_happy_path(self) -> None:
        """
        Tests the beta (1/Hartree) -> T (Kelvin) conversion.
        """
        beta_au = 1000.0  # An arbitrary internal beta value
        expected_temp = 1.0 / (1000.0 * KB_AU_HARTREE_PER_KELVIN)

        assert units.Temperature.to_kelvin(beta_au) == pytest.approx(expected_temp)

    @pytest.mark.parametrize("temp_k", [1.0, 77.0, 298.15, 300.0, 1000.0])
    def test_roundtrip_kelvin_to_beta_to_kelvin(self, temp_k: float) -> None:
        """
        Ensures that T -> beta -> T returns the original temperature.
        This is the most robust check.
        """
        # We do not use our manually calculated constant here.
        # We test the class against itself.
        beta = units.Temperature.to_beta(temp_k)
        result_temp = units.Temperature.to_kelvin(beta)

        assert result_temp == pytest.approx(temp_k)

    @pytest.mark.parametrize("beta_au", [100.0, 1052.79, 5000.0])
    def test_roundtrip_beta_to_kelvin_to_beta(self, beta_au: float) -> None:
        """
        Ensures that beta -> T -> beta returns the original beta.
        """
        temp = units.Temperature.to_kelvin(beta_au)
        result_beta = units.Temperature.to_beta(temp)

        assert result_beta == pytest.approx(beta_au)

    @pytest.mark.parametrize("invalid_temp", [0.0, -100.0, -0.01])
    def test_to_beta_raises_error_on_non_positive_temp(
        self, invalid_temp: float
    ) -> None:
        """
        Tests that T <= 0 raises a ValueError, as required
        by the physical formula.
        """
        with pytest.raises(ValueError, match="Temperature must be positive"):
            units.Temperature.to_beta(invalid_temp)

    @pytest.mark.parametrize("invalid_beta", [0.0, -1000.0, -1e-10])
    def test_to_kelvin_raises_error_on_non_positive_beta(
        self, invalid_beta: float
    ) -> None:
        """
        Tests that beta <= 0 raises a ValueError.
        """
        with pytest.raises(ValueError, match="Beta must be positive"):
            units.Temperature.to_kelvin(invalid_beta)
