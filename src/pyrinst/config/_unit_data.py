"""
Internal data module for unit conversion factors.

This module stores the central registry of all conversion factors to SI units.
It is not intended for direct use by end-users.
"""

import collections.abc
from scipy import constants as sc


class CaseInsensitiveDict(collections.abc.Mapping):
    """
    A dictionary-like object where key access is case-insensitive.

    Upon initialization, it converts all keys in the input dictionary
    to lowercase. When an item is accessed (e.g., my_dict['KEY']),
    it automatically converts the access key to lowercase before lookup.
    """
    def __init__(self, data):
        self._data = {k.lower(): v for k, v in data.items()}

    def __getitem__(self, key):
        return self._data[key.lower()]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)


# A private, centralized registry for all unit conversion factors to SI.
# This makes maintenance and extension much simpler.
_RAW_UNIT_REGISTRY = {
    'energy': {
        'J': 1.0,
        'joule': 1.0,
        'Eh': sc.value("Hartree energy"),
        'hartree': sc.value("Hartree energy"),
        'eV': sc.e,
        'kJ/mol': 1000.0 / sc.N_A,
        'kcal/mol': sc.calorie * 1000.0 / sc.N_A,
        'cm-1': sc.h * sc.c * 100.0,
        'K': sc.k,
        'Hz': sc.h,
        'erg': sc.erg,
        'meV': sc.e * 1e-3,
        'au': sc.value("Hartree energy"),
    },
    'length': {
        'm': 1.0,
        'metre': 1.0,
        'A': sc.angstrom,
        'angstrom': sc.angstrom,
        'a0': sc.value("Bohr radius"),
        'bohr': sc.value("Bohr radius"),
        'mm': 1e-3,
        'cm': 1e-2,
        'in': 2.54e-2,
        'au': sc.value("Bohr radius"),
    },
    'mass': {
        'kg': 1.0,
        'g': 1e-3,
        'amu': sc.m_u,
        'Da': sc.m_u,  # Dalton is the modern term for amu
        'au': sc.m_e,
    },
    'time': {
        's': 1.0,
        'fs': sc.femto,
        'ps': sc.pico,
        'ns': sc.nano,
        'mus': sc.micro,
        'au': sc.value("atomic unit of time"),
    },
}

# Add frequency aliases to energy
for freq_unit in ['kHz', 'MHz', 'GHz']:
    multiplier = {'k': 1e3, 'M': 1e6, 'G': 1e9}[freq_unit[0]]
    _RAW_UNIT_REGISTRY['energy'][freq_unit] = _RAW_UNIT_REGISTRY['energy']['Hz'] * multiplier

# wrap each sub-dictionary in CaseInsensitiveDict
UNIT_REGISTRY = CaseInsensitiveDict({
    quantity_type: CaseInsensitiveDict(units)
    for quantity_type, units in _RAW_UNIT_REGISTRY.items()
})
