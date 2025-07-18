import json
import logging
import re
from importlib import resources
from typing import Any, Callable, Dict, Optional, Tuple, Union

# Get a module-level logger with the correct package name.
logger = logging.getLogger(__name__)


class ElementData:
    """Handles loading and querying of atomic data for AIMD simulations.

    This class provides a simple interface to get atomic mass and atomic number
    based on an element or isotope symbol (e.g., 'C', 'C13', 'D'). It is
    designed to be instantiated once at the module level to create a shared,
    efficient data provider.
    """

    def __init__(self, data_path: str = 'atomic_data.json') -> None:
        """Initializes the ElementData object by loading data from a JSON file."""
        self._elements: Dict[str, Any] = {}
        self._aliases: Dict[str, Any] = {}
        try:
            self._load_data(data_path)
            print(self._elements)
        except (FileNotFoundError, ModuleNotFoundError) as e:
            # This allows the instance to be created even if the package is not
            # installed, for cases where data is mocked later (like in our main block).
            logger.error(
                f"Could not load file {data_path}. The Element instance will not be created. "
            )
            raise e

    def _load_data(self, data_path: str) -> None:
        """Loads and parses the atomic data from the specified JSON file."""
        logger.info("Attempting to load atomic data via package resources...")
        try:
            # This is the robust way to access package data.
            # It requires the package to be installed (e.g., `pip install -e .`).
            file_resource = (
                resources.files('easy_instanton.config').joinpath(data_path)
            )
            logger.info("Loading data from: %s", file_resource)
            with file_resource.open('r', encoding='utf-8') as f:
                data = json.load(f)
        except (FileNotFoundError, ModuleNotFoundError) as e:
            logger.error(
                "Could not find package 'easy_instanton.config'. "
                "Is the package installed correctly?",
            )
            raise e  # Re-raise the exception

        self._elements = data.get('elements', {})
        self._aliases = data.get('isotope_aliases', {})
        logger.info(
            "Successfully loaded data for %d elements and %d aliases.",
            len(self._elements),
            len(self._aliases),
        )
    
    def _parse_symbol(self, symbol: str) -> Tuple[str, Optional[str]]:
        """Parses a symbol to determine the base element and mass number.

        Parameters
        ----------
        symbol : str
            The input symbol (e.g., 'C', 'C13', 'D').

        Returns
        -------
        tuple[str, str | None]
            A tuple containing (base_element_symbol, mass_number_str).
            The mass number is None if the symbol represents a generic element.
            Example:
            - 'C13' -> ('C', '13')
            - 'C'   -> ('C', None)
            - 'D'   -> ('H', '2')

        Raises
        ------
        KeyError
            If the symbol cannot be parsed into a valid element or isotope.
        """
        logger.debug("Parsing symbol: '%s'", symbol)
        # 1. Check if the symbol is a special alias like 'D' or 'T'
        if symbol in self._aliases:
            alias_info = self._aliases[symbol]
            element = alias_info['element']
            mass_number = str(alias_info['massNumber'])
            logger.debug(
                "Symbol '%s' found in aliases. Mapping to element '%s', mass number '%s'.",
                symbol,
                element,
                mass_number,
            )
            return element, mass_number

        # 2. Check if the symbol specifies an isotope, e.g., 'C13'
        match = re.match(r'^([A-Za-z]+)([0-9]+)$', symbol)
        if match:
            element, mass_number = match.groups()
            if element in self._elements:
                logger.debug(
                    "Symbol '%s' matched isotope pattern. Element: '%s', mass number: '%s'.",
                    symbol,
                    element,
                    mass_number,
                )
                return element, mass_number

        # 3. Assume it's a base element symbol, e.g., 'C'
        if symbol in self._elements:
            logger.debug("Symbol '%s' is a base element.", symbol)
            return symbol, None

        logger.error("Symbol '%s' is not a valid element or known isotope.", symbol)
        raise KeyError(f"Symbol '{symbol}' is not a valid element or isotope alias.")

    def get_mass(self, symbol: str) -> float:
        """Gets the atomic mass for a given symbol.

        If a generic symbol (e.g., 'C') is provided, this method returns the
        conventional atomic weight. If a specific isotope symbol (e.g., 'C13'
        or 'D') is provided, it returns the mass of that specific isotope.

        Parameters
        ----------
        symbol : str
            The atomic symbol to look up (e.g., 'H', 'D', 'C13').

        Returns
        -------
        float
            The requested atomic mass in atomic mass units (u).

        Raises
        ------
        KeyError
            If the symbol or its corresponding isotope is not found.

        Examples
        --------
        >>> from easy_instanton.utils.elements import element_data
        >>> # Get conventional mass
        >>> element_data.get_mass('C')
        12.011
        >>> # Get specific isotope mass
        >>> element_data.get_mass('D')
        2.01410177812
        """
        element, mass_number = self._parse_symbol(symbol)

        if mass_number:
            # It's an isotope, get the isotope mass
            try:
                mass = self._elements[element]['isotopes'][mass_number]['mass']
                logger.debug("Found isotope mass for '%s': %f", symbol, mass)
                return mass
            except KeyError:
                logger.error(
                    "Isotope for symbol '%s' (Element: %s, Mass Number: %s) not in database.",
                    symbol,
                    element,
                    mass_number,
                )
                raise KeyError(f"Isotope '{symbol}' not found in the database.")
        else:
            # It's a generic element, get the conventional mass
            mass = self._elements[element]['mass']
            logger.debug("Found conventional mass for '%s': %f", symbol, mass)
            return mass

    def get_atomic_number(self, symbol: str) -> int:
        """Gets the atomic number for any valid symbol.

        Parameters
        ----------
        symbol : str
            The atomic symbol to look up (e.g., 'H', 'D', 'C13').

        Returns
        -------
        int
            The atomic number (Z) of the element.

        Raises
        ------
        KeyError
            If the symbol cannot be resolved to a known element.

        Examples
        --------
        >>> from your_package_name.utils.elements import element_data
        >>> element_data.get_atomic_number('C')
        6
        >>> element_data.get_atomic_number('C13')
        6
        >>> element_data.get_atomic_number('D')
        1
        """
        element, _ = self._parse_symbol(symbol)
        atomic_num = self._elements[element]['atomicNumber']
        logger.debug(
            "Found atomic number for '%s' (Element: %s): %d",
            symbol,
            element,
            atomic_num,
        )
        return atomic_num


# Create a single, shared instance of the class at the module level.
# This will be executed only once when the module is first imported,
# making it highly efficient.
element_data = ElementData()

if __name__ == '__main__':
    # This block only runs when you execute `python elements.py` directly.

    def setup_for_demonstration():
        """
        A helper to load mock data into the element_data instance.
        This makes our script self-contained and runnable without a real JSON file.
        """
        print("--- Setting up mock data for demonstration ---")
        mock_data = {
            "elements": {
                "H": {"atomicNumber": 1, "mass": 1.008, "isotopes": {
                    "1": {"mass": 1.007825}, "2": {"mass": 2.014102}
                }},
                "C": {"atomicNumber": 6, "mass": 12.011, "isotopes": {
                    "12": {"mass": 12.0}, "13": {"mass": 13.003355}
                }}
            },
            "isotope_aliases": {"D": {"element": "H", "massNumber": 2}}
        }
        element_data._elements = mock_data['elements']
        element_data._aliases = mock_data['isotope_aliases']

    def demonstrate_mass_queries():
        """Shows how to get different types of masses."""
        print("\n--- Demonstrating Mass Queries ---")

        symbol = 'C'
        mass = element_data.get_mass(symbol)
        print(f"Input: '{symbol}' -> Conventional Mass: {mass}")

        symbol = 'D'
        mass = element_data.get_mass(symbol)
        print(f"Input: '{symbol}' -> Isotope Mass (from alias): {mass}")

        symbol = 'C13'
        mass = element_data.get_mass(symbol)
        print(f"Input: '{symbol}' -> Isotope Mass (from pattern): {mass}")

    def demonstrate_atomic_number_queries():
        """Shows how to get atomic numbers."""
        print("\n--- Demonstrating Atomic Number Queries ---")

        symbol = 'C'
        z = element_data.get_atomic_number(symbol)
        print(f"Input: '{symbol}' -> Atomic Number: {z}")

        symbol = 'D'
        z = element_data.get_atomic_number(symbol)
        print(f"Input: '{symbol}' -> Atomic Number: {z}")

    def demonstrate_error_handling():
        """Shows how the class handles invalid input."""
        print("\n--- Demonstrating Error Handling ---")
        invalid_symbol = 'Xyz'
        print(f"Testing invalid symbol: '{invalid_symbol}'")
        try:
            element_data.get_mass(invalid_symbol)
        except KeyError as e:
            # The 'SUCCESS' here means we successfully caught the error we expected.
            print(f"SUCCESS: Correctly caught expected error -> {e}")

    def main():
        """The main entry point for the demonstration."""
        setup_for_demonstration()
        print("\n>>> Starting simple demonstrations for ElementData <<<")
        demonstrate_mass_queries()
        demonstrate_atomic_number_queries()
        demonstrate_error_handling()
        print("\n>>> Demonstration Finished <<<")

    # Run the main demonstration function
    main()
