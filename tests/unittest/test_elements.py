# tests/test_elements.py

import logging
from typing import Any

import numpy as np
import pytest

from pyrinst.utils.elements import element_data

# Define mock data at the module level for clarity and reuse.
MOCK_DATA: dict[str, Any] = {
    "elements": {
        "H": {
            "atomicNumber": 1,
            "mass": 1.008,
            "isotopes": {
            "mu": {"mass": 0.113977478, "composition": -1.0},
            "1": {"mass": 1.00782503223, "composition": 0.999885},
            "2": {"mass": 2.01410177812, "composition": 0.000115},
            "3": {"mass": 3.0160492779, "composition": -1.0}
            }
        },
        "C": {
            "atomicNumber": 6,
            "mass": 12.011,
            "isotopes": {
                "12": {"mass": 12.0, "composition": 0.9893},
                "13": {"mass": 13.00335483507, "composition": 0.0107},
                "14": {"mass": 14.0032419884, "composition": -1.0}
            }
        },
        "Tc": {
            "name": "Technetium",
            "atomicNumber": 43,
            "mass": 96.90636,
            "isotopes": {
                "97": {"mass": 96.9063667, "composition": -1.0},
                "98": {"mass": 97.9072124, "composition": -1.0},
                "99": {"mass": 98.9062508, "composition": -1.0}
            }
    },
    }, # elements
    "isotope_aliases": {"D": {"element": "H", "massNumber": 2}},
}


@pytest.fixture
def mocked_element_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pytest fixture to mock the data within the element_data instance.

    This fixture uses `monkeypatch` to temporarily replace the internal data
    dictionaries of the `element_data` singleton instance for the duration
    of a test. This ensures that tests are isolated, repeatable, and do not
    depend on the external JSON file.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        A built-in pytest fixture for safely modifying classes, methods,
        or variables.
    """
    monkeypatch.setattr(element_data, "_elements", MOCK_DATA["elements"])
    monkeypatch.setattr(element_data, "_aliases", MOCK_DATA["isotope_aliases"])


@pytest.mark.parametrize(
    "symbol, expected_mass",
    [
        # Case 1: Stable element, should use most abundant isotope's mass
        ("C", 12.0),
        ("H", 1.00782503223),

        # Case 2: Unstable element, should use fallback top-level mass
        ("Tc", 96.90636),

        # Case 3: Specific isotope via alias
        ("D", 2.01410177812),

        # Case 4: Specific isotope via pattern
        ("C13", 13.00335483507),
        ("Tc98", 97.9072124), # Also test specific unstable isotope
    ],
)
def test_get_mass_happy_paths(
    mocked_element_data: None, symbol: str, expected_mass: float
) -> None:
    """Tests successful retrieval of various masses.

    This test is parameterized to cover multiple valid scenarios efficiently,
    including lookups for conventional mass, alias-based isotopes, and
    pattern-based isotopes.

    Parameters
    ----------
    mocked_element_data : None
        This argument activates the fixture to load mock data.
    symbol : str
        The atomic symbol to be tested.
    expected_mass : float
        The expected mass for the given symbol.
    """
    actual_mass = element_data.get_mass(symbol)
    assert abs(actual_mass - expected_mass) < 1e-9, (
        f"Mass for symbol '{symbol}' did not match expected value."
    )


@pytest.mark.parametrize(
    "symbol, expected_z",
    [
        ("C", 6),
        ("C13", 6),
        ("H", 1),
        ("D", 1),
    ],
)
def test_get_atomic_number_happy_paths(
    mocked_element_data: None, symbol: str, expected_z: int
) -> None:
    """Tests successful retrieval of atomic numbers.

    This test is parameterized to verify that various symbol types all
    correctly resolve to the same, correct atomic number for an element.

    Parameters
    ----------
    mocked_element_data : None
        This argument activates the fixture to load mock data.
    symbol : str
        The atomic symbol to be tested.
    expected_z : int
        The expected atomic number for the given symbol.
    """
    actual_z = element_data.get_atomic_number(symbol)
    assert actual_z == expected_z, (
        f"Atomic number for symbol '{symbol}' did not match expected value."
    )


@pytest.mark.parametrize(
    "invalid_symbol",
    [
        "Xyz",  # Completely invalid element
        "C99",  # Valid element, but non-existent isotope
        "d",  # Incorrect case for alias
    ],
)
def test_get_mass_raises_keyerror_for_invalid_symbols(
    mocked_element_data: None, invalid_symbol: str
) -> None:
    """Verifies that KeyError is raised for various invalid symbols.

    This parameterized test ensures that the system is robust against different
    types of incorrect user input.

    Parameters
    ----------
    mocked_element_data : None
        This argument activates the fixture to load mock data.
    invalid_symbol : str
        The invalid symbol that is expected to cause a KeyError.
    """
    with pytest.raises(KeyError):
        element_data.get_mass(invalid_symbol)


def test_logging_for_invalid_symbol(
    mocked_element_data: None, caplog: pytest.LogCaptureFixture
) -> None:
    """Tests that an ERROR message is logged for an invalid symbol.

    This test demonstrates how to use the `caplog` fixture to inspect
    log messages generated by the code under test.

    Parameters
    ----------
    mocked_element_data : None
        This argument activates the fixture to load mock data.
    caplog : pytest.LogCaptureFixture
        A built-in pytest fixture that captures logging output.
    """
    with caplog.at_level(logging.ERROR), pytest.raises(KeyError):
        element_data.get_mass("Xyz")

    # Assert that the specific error message we expect is present in the logs.
    assert "not a valid element" in caplog.text


@pytest.mark.parametrize(
    "input_symbol, expected_base_symbol",
    [
        ("H", "H"),       # Base symbol -> Base symbol
        ("C", "C"),       # Base symbol -> Base symbol
        ("D", "H"),       # Alias -> Base symbol
        ("C13", "C"),     # Pattern -> Base symbol
        ("H1", "H"),      # Pattern -> Base symbol
    ]
)
def test_get_base_symbol(input_symbol: str, expected_base_symbol: str):
    """Tests that any valid symbol can be resolved to its base element symbol.

    This parameterized test covers all cases: base symbols, aliases, and
    pattern-based isotopes.

    Parameters
    ----------
    input_symbol : str
        The symbol to be normalized.
    expected_base_symbol : str
        The expected fundamental element symbol.
    """
    actual_base_symbol = element_data.get_base_symbol(input_symbol)
    assert actual_base_symbol == expected_base_symbol


def test_get_base_symbol_raises_keyerror():
    """Tests that get_base_symbol raises KeyError for invalid input."""
    with pytest.raises(KeyError, match="not a valid element"):
        element_data.get_base_symbol("Xyz")


@pytest.mark.parametrize(
    "atomic_number, expected_symbol",
    [
        (1, "H"),   # Hydrogen
        (6, "C"),   # Carbon
        (43, "Tc"), # Technetium
    ],
)
def test_get_symbol_happy_paths(
    mocked_element_data: None, atomic_number: int, expected_symbol: str
) -> None:
    """Tests successful retrieval of element symbols from atomic numbers.

    This parameterized test verifies that the get_symbol method correctly
    converts atomic numbers to their corresponding element symbols.

    Parameters
    ----------
    mocked_element_data : None
        This argument activates the fixture to load mock data.
    atomic_number : int
        The atomic number to be tested.
    expected_symbol : str
        The expected element symbol for the given atomic number.
    """
    actual_symbol = element_data.get_symbol(atomic_number)
    assert actual_symbol == expected_symbol, (
        f"Symbol for atomic number '{atomic_number}' did not match expected value."
    )


@pytest.mark.parametrize(
    "invalid_atomic_number",
    [
        0,    # Invalid: atomic number must be positive
        999,  # Invalid: not in database
        -1,   # Invalid: negative atomic number
    ],
)
def test_get_symbol_raises_keyerror_for_invalid_numbers(
    mocked_element_data: None, invalid_atomic_number: int
) -> None:
    """Verifies that KeyError is raised for invalid atomic numbers.

    This parameterized test ensures that the system is robust against
    different types of incorrect atomic number input.

    Parameters
    ----------
    mocked_element_data : None
        This argument activates the fixture to load mock data.
    invalid_atomic_number : int
        The invalid atomic number that is expected to cause a KeyError.
    """
    with pytest.raises(KeyError):
        element_data.get_symbol(invalid_atomic_number)


def test_logging_for_invalid_atomic_number(
    mocked_element_data: None, caplog: pytest.LogCaptureFixture
) -> None:
    """Tests that an ERROR message is logged for an invalid atomic number.

    This test demonstrates logging inspection for the get_symbol method.

    Parameters
    ----------
    mocked_element_data : None
        This argument activates the fixture to load mock data.
    caplog : pytest.LogCaptureFixture
        A built-in pytest fixture that captures logging output.
    """
    with caplog.at_level(logging.ERROR), pytest.raises(KeyError):
        element_data.get_symbol(999)

    # Assert that the specific error message we expect is present in the logs.
    assert "not found in database" in caplog.text


# --- TESTS FOR BATCH-PROCESSING METHODS ---

def test_get_masses_batch():
    """Tests the batch processing of get_masses."""
    symbols = ['C', 'D', 'C13', 'H']
    expected_masses = np.array([12.0, 2.014102, 13.003355, 1.007825])

    actual_masses = element_data.get_masses(symbols)

    assert isinstance(actual_masses, np.ndarray)
    assert np.allclose(actual_masses, expected_masses)


def test_get_atomic_numbers_batch():
    """Tests the batch processing of get_atomic_numbers."""
    symbols = ['C', 'D', 'C13', 'H']
    expected_numbers = np.array([6, 1, 6, 1])

    actual_numbers = element_data.get_atomic_numbers(symbols)

    assert isinstance(actual_numbers, np.ndarray)
    assert np.array_equal(actual_numbers, expected_numbers)


def test_get_base_symbols_batch():
    """Tests the batch processing of get_base_symbols."""
    symbols = ['C', 'D', 'C13', 'H']
    expected_symbols = np.array(['C', 'H', 'C', 'H'])

    actual_symbols = element_data.get_base_symbols(symbols)

    assert isinstance(actual_symbols, np.ndarray)
    assert np.array_equal(actual_symbols, expected_symbols)


def test_get_symbols_batch():
    """Tests the batch processing of get_symbols."""
    atomic_numbers = [1, 1, 8, 6]
    expected_symbols = np.array(['H', 'H', 'O', 'C'])

    actual_symbols = element_data.get_symbols(atomic_numbers)

    assert isinstance(actual_symbols, np.ndarray)
    assert np.array_equal(actual_symbols, expected_symbols)


def test_batch_functions_with_empty_list():
    """Tests that batch functions return empty arrays for empty input."""
    assert element_data.get_masses([]).shape == (0,)
    assert element_data.get_atomic_numbers([]).shape == (0,)
    assert element_data.get_base_symbols([]).shape == (0,)
    assert element_data.get_symbols([]).shape == (0,)
