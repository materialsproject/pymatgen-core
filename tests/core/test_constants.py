"""Parity tests for pymatgen.core.constants vs scipy.constants.

pymatgen.core.constants embeds CODATA values as Python literals to avoid the
~50 ms import cost of scipy.constants. These tests assert byte-exact agreement
with the currently installed scipy. When scipy bumps to a new CODATA release,
these tests will fail and prompt an update of the embedded literals.
"""

from __future__ import annotations

import pytest
import scipy.constants as sc

from pymatgen.core import constants as pmg_const

# Top-level named constants we expose. Each is compared to the same-named
# attribute on scipy.constants.
NAMED_CONSTANTS = [
    "e",
    "N_A",
    "Avogadro",
    "Boltzmann",
    "k",
    "R",
    "hbar",
    "h",
    "c",
    "m_e",
    "epsilon_0",
    "mile",
    "calorie",
    "tera",
    "milli",
    "centi",
]


@pytest.mark.parametrize("name", NAMED_CONSTANTS)
def test_named_constant_matches_scipy(name: str) -> None:
    pmg_val = getattr(pmg_const, name)
    scipy_val = getattr(sc, name)
    assert pmg_val == scipy_val, f"{name}: pymatgen={pmg_val!r}  scipy={scipy_val!r}"


@pytest.mark.parametrize("key", list(pmg_const.physical_constants))
def test_physical_constant_matches_scipy(key: str) -> None:
    """Every key in pymatgen's physical_constants must match scipy exactly."""
    pmg_tuple = pmg_const.physical_constants[key]
    scipy_tuple = sc.physical_constants[key]
    assert pmg_tuple == scipy_tuple, f"{key}: pymatgen={pmg_tuple!r}  scipy={scipy_tuple!r}"


@pytest.mark.parametrize("key", list(pmg_const.physical_constants))
def test_value_helper_matches_scipy(key: str) -> None:
    """``pmg_const.value(key)`` must match ``scipy.constants.value(key)``."""
    assert pmg_const.value(key) == sc.value(key), f"value({key!r}) mismatch"


def test_no_scipy_constants_on_import_path() -> None:
    """Importing pymatgen.core.constants must not pull in scipy.constants.

    The whole point of this module is to avoid the scipy.constants subtree
    (which drags in numpy.f2py / fileinput / charset_normalizer). This test
    re-imports both in a fresh sys.modules-pruned subprocess to verify.
    """
    import subprocess
    import sys

    code = (
        "import sys; "
        "from pymatgen.core import constants; "
        "assert 'scipy.constants' not in sys.modules, "
        "    f'scipy.constants leaked: {sorted(m for m in sys.modules if m.startswith(\"scipy\"))!r}'; "
        "print('OK')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
