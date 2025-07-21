"""
Integration tests for the Easy Instanton package.

These tests verify that the components of the package, such as the code
and its packaged data files, work together correctly in an installed
environment. They do NOT use mock data.
"""

import pytest

from easy_instanton.utils.elements import element_data


def test_data_is_loaded():
    """
    Tests if the element data was loaded from the JSON file.

    This is a basic "smoke test" to ensure that the `element_data` instance
    is not empty after being imported, which implies that the `_load_data`
    method successfully found and parsed the `atomic_data.json` file.
    """
    # The _elements dictionary should not be empty if data loading succeeded.
    assert element_data._elements, "Atomic data dictionary should not be empty."
    assert element_data._aliases is not None, "Aliases dictionary should exist."


def test_spot_check_real_data_mass():
    """
    Performs a spot check on a few key values from the real data file.

    This test implicitly verifies the entire chain:
    1. The package was installed correctly.
    2. The `atomic_data.json` file was included in the installation.
    3. The file path resolution in `_load_data` works.
    4. The JSON is valid and was parsed correctly.
    5. The `get_mass` method can retrieve data from the loaded structure.
    """
    # These values should correspond exactly to what's in your JSON file.
    # We test one conventional mass and one alias.
    assert abs(element_data.get_mass('H') - 1.00782503223) < 1e-9
    assert abs(element_data.get_mass('D') - 2.014101778) < 1e-9


def test_spot_check_real_data_atomic_number():
    """Performs a spot check on atomic number lookups with real data."""
    assert element_data.get_atomic_number('C') == 6
    assert element_data.get_atomic_number('D') == 1


def test_nonexistent_real_isotope_raises_error():
    """
    Ensures that querying a non-existent isotope from the real data
    correctly raises a KeyError.
    """
    with pytest.raises(KeyError, match="Isotope 'C99' not found"):
        element_data.get_mass('C99')