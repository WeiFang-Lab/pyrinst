"""
Internal Physical Constants for PyRInst.

This module defines the fixed, internal unit system used by all core
computation routines within `pyrinst.core`.

Internal Standard: Atomic Units (AU)
------------------------------------
All core calculations (optimization, path integrals, Hessians) MUST
operate in a consistent Atomic Unit (AU) system, defined as:
- Energy: Hartree
- Length: Bohr
- Mass:   m_e (electron mass)
- hbar:   1.0

Purpose
-------
This file provides the single source of truth for this internal standard
and, crucially, provides the conversion factors to bridge this standard
with common "external" units (e.g., Angstrom, kcal/mol, amu) that are
used in I/O operations (like `main.py`) or in defining a PES (like
`custom_pes.py`).

These constants are calculated *once* using the `units.py` toolkit
and should be imported by any module that performs I/O.
"""

# Use the unit conversion toolkit to calculate factors
from pyrinst.utils.units import Energy, Length, Mass, Time

# --- 1. Internal Standard Base Units (AU) ---
# These are the reference units for all internal calculations.
HARTREE: float = 1.0
BOHR: float = 1.0
M_E: float = 1.0       # Electron mass ('au' in _unit_data.py)
AU_TIME: float = 1.0
HBAR: float = 1.0


# --- 2. External-to-Internal Conversion Factors ---
# Factors to convert common EXTERNAL units TO the INTERNAL AU standard.

# Energy: External -> Hartree
KCAL_MOL: float = Energy(1.0, 'kcal/mol').get('Hartree')
EV: float = Energy(1.0, 'eV').get('Hartree')
KELVIN: float = Energy(1.0, 'K').get('Hartree')

# Length: External -> Bohr
ANGSTROM: float = Length(1.0, 'A').get('Bohr')

# Mass: External -> m_e (electron mass)
AMU: float = Mass(1.0, 'amu').get('au')  # 'au' is m_e

# Time: External -> au_time
FEMTOSECOND: float = Time(1.0, 'fs').get('au')


# --- 3. Derived Physical Constants (in Internal AU) ---

# Boltzmann constant in (Hartree / K)
KB: float = KELVIN