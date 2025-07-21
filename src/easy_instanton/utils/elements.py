import json
import logging
import re
from importlib import resources
from typing import Any, Callable, Dict, Optional, Tuple, Union, Iterable

import numpy as np

# Get a module-level logger with the correct package name.
logger = logging.getLogger(__name__)


class ElementData:
    """Handles loading and querying of atomic data for PyRInst.

    This class provides a simple interface to get atomic mass and atomic number
    based on an element or isotope symbol. When a generic element symbol is
    given (e.g., 'C'), it automatically uses the mass of the most abundant
    stable isotope.
    """

    def __init__(self, data_path: str = 'atomic_data.json') -> None:
        """Initializes the ElementData object by loading data from a JSON file."""
        self._elements: Dict[str, Any] = {}
        self._aliases: Dict[str, Any] = {}
        # This new dictionary will store the most abundant isotope for each element.
        self._default_isotopes: Dict[str, str] = {}
        try:
            self._load_data(data_path)
            self._preprocess_data()  # Pre-calculate defaults after loading
        except (FileNotFoundError, ModuleNotFoundError) as e:
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
            raise e

        self._elements = data.get('elements', {})
        self._aliases = data.get('isotope_aliases', {})
        logger.info(
            "Successfully loaded data for %d elements and %d aliases.",
            len(self._elements),
            len(self._aliases),
        )
    
    def _preprocess_data(self) -> None:
        """
        Pre-processes the loaded data to find the most abundant isotope
        for each element, which will be used as the default.
        """
        logger.info("Preprocessing data to determine default isotopes.")
        
        self._default_isotopes = {}

        for symbol, data in self._elements.items():
            isotopes = data.get('isotopes', {})
            if not isotopes:
                logger.warning("Element '%s' has no isotope data.", symbol)
                continue

            most_abundant_isotope = None
            max_composition = -1.0

            for mass_number, isotope_info in isotopes.items():
                composition = isotope_info.get('composition', -1.0)
                # We only consider isotopes with a valid, non-negative composition.
                if composition > max_composition:
                    max_composition = composition
                    most_abundant_isotope = mass_number

            if most_abundant_isotope:
                self._default_isotopes[symbol] = most_abundant_isotope
                logger.debug(
                    "Default isotope for '%s' set to mass number '%s' "
                    "(composition: %f).",
                    symbol, most_abundant_isotope, max_composition
                )
        logger.debug("Atomic data preprocessing complete.")
    
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

        if mass_number is None:
            if element not in self._default_isotopes:
                raise KeyError(
                    f"No default isotope could be determined for element '{element}'."
                )
            mass_number = self._default_isotopes[element]
            logger.debug(
                "No isotope specified for '%s'; using most abundant: '%s'.",
                element, mass_number
            )

        try:
            mass = self._elements[element]['isotopes'][mass_number]['mass']
            logger.debug("Found isotope mass for '%s-%s': %f", element, mass_number, mass)
            return mass
        except KeyError:
            logger.error(
                "Isotope '%s-%s' for symbol '%s' not in database.",
                element, mass_number, symbol
            )
            raise KeyError(f"Isotope '{symbol}' not found in the database.")

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

    def get_base_symbol(self, symbol: str) -> str:
        """Returns the base element symbol for any given isotope symbol.

        This method normalizes an isotope-specific symbol (like 'D' or 'C13')
        to its fundamental element symbol (like 'H' or 'C'). This is useful
        for electronic structure calculations where only the element's identity
        is needed, not its mass.

        Parameters
        ----------
        symbol : str
            The atomic or isotope symbol to look up (e.g., 'H', 'D', 'C13').

        Returns
        -------
        str
            The corresponding base element symbol (e.g., 'H', 'C').

        Raises
        ------
        KeyError
            If the symbol cannot be resolved to a known element.

        Examples
        --------
        >>> from easy_instanton.utils.elements import element_data
        >>> element_data.get_base_symbol('D')
        'H'
        >>> element_data.get_base_symbol('C13')
        'C'
        >>> element_data.get_base_symbol('H')
        'H'
        """
        element, _ = self._parse_symbol(symbol)
        logger.debug(
            "Resolved symbol '%s' to base element '%s'", symbol, element
        )
        return element

    def get_masses(self, symbols: Iterable[str]) -> np.ndarray:
        """Gets atomic masses for a list or array of symbols.

        This is a vectorized version of `get_mass` for efficient batch
        processing.

        Parameters
        ----------
        symbols : Iterable[str]
            An iterable (e.g., list, tuple, or np.ndarray) of atomic symbols.

        Returns
        -------
        np.ndarray
            A NumPy array of atomic masses, with the same length as the input.

        Examples
        --------
        >>> from easy_instanton.utils.elements import element_data
        >>> symbols = ['C', 'D', 'C13']
        >>> element_data.get_masses(symbols)
        array([12.0, 2.014102, 13.003355])
        """
        # The `otypes` argument helps NumPy pre-allocate an array of the correct type.
        vectorized_func = np.vectorize(self.get_mass, otypes=[float])
        return vectorized_func(symbols)

    def get_atomic_numbers(self, symbols: Iterable[str]) -> np.ndarray:
        """Gets atomic numbers for a list or array of symbols.

        This is a vectorized version of `get_atomic_number`.

        Parameters
        ----------
        symbols : Iterable[str]
            An iterable of atomic symbols.

        Returns
        -------
        np.ndarray
            A NumPy array of atomic numbers.
        """
        vectorized_func = np.vectorize(self.get_atomic_number, otypes=[int])
        return vectorized_func(symbols)

    def get_base_symbols(self, symbols: Iterable[str]) -> np.ndarray:
        """Gets base element symbols for a list or array of symbols.

        This is a vectorized version of `get_base_symbol`.

        Parameters
        ----------
        symbols : Iterable[str]
            An iterable of atomic or isotope symbols.

        Returns
        -------
        np.ndarray
            A NumPy array of base element symbols.
        """
        # For strings, it's common to specify the output type as object and
        # then let NumPy infer the final string type, or use a specific
        # string dtype like '<U2' (a 2-character unicode string).
        vectorized_func = np.vectorize(self.get_base_symbol, otypes=[object])
        return vectorized_func(symbols)

# Create a single, shared instance of the class at the module level.
# This will be executed only once when the module is first imported,
# making it highly efficient.
element_data = ElementData()
