"""
CODATA physical constants used throughout pymatgen-core.

Drop-in replacement for the subset of ``scipy.constants`` that pymatgen-core
consumes. Importing ``scipy.constants`` is expensive (~50 ms on cold import,
because it pulls ``scipy._lib.array_api_compat`` → ``numpy.f2py.crackfortran``
→ ``fileinput`` → ``charset_normalizer``); this module avoids that cost
entirely by embedding CODATA values as Python literals.

Values mirror ``scipy.constants`` (CODATA 2022 / SI 2019 redefinition). The
unit test in ``tests/core/test_constants.py`` asserts byte-exact parity with
the currently installed scipy, so any future CODATA revision shipped by
scipy will fail CI and prompt an update here.

Public API mirrors the parts of ``scipy.constants`` pymatgen-core actually
uses:

- module-level named constants: ``e``, ``N_A``, ``Avogadro``, ``Boltzmann``,
  ``k``, ``R``, ``hbar``, ``h``, ``c``, ``m_e``, ``epsilon_0``, ``mile``,
  ``calorie``, ``tera``, ``milli``, ``centi``
- ``physical_constants``: dict[str, tuple[value, unit, uncertainty]]
- ``value(name)``: helper that returns ``physical_constants[name][0]``
"""

from __future__ import annotations

from typing import Any

# Defining constants (exact, by SI 2019 redefinition)
e: float = 1.602176634e-19  # elementary charge, C
N_A: float = 6.02214076e23  # Avogadro number, mol^-1
Avogadro: float = N_A
Boltzmann: float = 1.380649e-23  # J K^-1
k: float = Boltzmann  # alias
h: float = 6.62607015e-34  # Planck constant, J s
hbar: float = 1.0545718176461565e-34  # reduced Planck, J s
c: float = 299792458.0  # speed of light, m s^-1

# Measured (CODATA 2022)
m_e: float = 9.1093837139e-31  # electron mass, kg
epsilon_0: float = 8.8541878188e-12  # vacuum permittivity, F m^-1
R: float = 8.31446261815324  # molar gas constant, J mol^-1 K^-1

# Unit conversion / SI prefixes
mile: float = 1609.3439999999998  # m
calorie: float = 4.184  # J (thermochemical)
tera: float = 1e12
milli: float = 1e-3
centi: float = 1e-2

# physical_constants — only the subset used by pymatgen-core.
# Structure mirrors scipy: (value, unit, std_uncertainty). The value is typed
# as Any (matching scipy's stubs) so downstream arithmetic with these returns
# Any and doesn't tighten previously loose typing.
physical_constants: dict[str, tuple[Any, str, Any]] = {
    "electron volt-hartree relationship": (0.036749322175665, "E_h", 4e-14),
    "atomic mass unit-kilogram relationship": (1.66053906892e-27, "kg", 5.2e-37),
    "Bohr radius": (5.29177210544e-11, "m", 8.2e-21),
    "Boltzmann constant in eV/K": (8.617333262145179e-05, "eV K^-1", 0.0),
    "atomic unit of length": (5.29177210544e-11, "m", 8.2e-21),
    "Rydberg constant times hc in eV": (13.60569312299, "eV", 1.5e-11),
    "Boltzmann constant in Hz/K": (20836619123.327576, "Hz K^-1", 0.0),
    "hertz-joule relationship": (6.62607015e-34, "J", 0.0),
    "hertz-electron volt relationship": (4.135667696923859e-15, "eV", 0.0),
    "hertz-hartree relationship": (1.5198298460574e-16, "E_h", 1.7e-28),
    "hertz-inverse meter relationship": (3.3356409519815204e-09, "m^-1", 0.0),
    "Angstrom star": (1.00001495e-10, "m", 9e-17),
    "Planck constant": (6.62607015e-34, "J Hz^-1", 0.0),
    "Boltzmann constant": (1.380649e-23, "J K^-1", 0.0),
}


def value(key: str) -> float:
    """Return ``physical_constants[key][0]``, matching ``scipy.constants.value``."""
    return physical_constants[key][0]
