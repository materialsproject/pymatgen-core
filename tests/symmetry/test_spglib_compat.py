from __future__ import annotations

import types
import warnings

import numpy as np
import pytest
import spglib

from pymatgen.core import Lattice, Structure
from pymatgen.symmetry._spglib_compat import SpglibError, call_spglib, ignore_spglib_error_handling_warning
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

OLD_ERROR_HANDLING_MSG = "Set OLD_ERROR_HANDLING to false"

# Rock salt NaCl: Fm-3m (225)
NACL_CELL = (
    np.eye(3) * 5.64,
    [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0.5]],
    [11, 11, 11, 11, 17, 17, 17, 17],
)
# Two atoms on top of each other: spglib cannot determine the symmetry
OVERLAPPING_CELL = (np.eye(3) * 3.0, [[0, 0, 0], [0, 0, 0]], [1, 1])


def _old_error_handling() -> bool | None:
    return getattr(getattr(spglib, "error", None), "OLD_ERROR_HANDLING", None)


@pytest.fixture
def no_spglib_deprecation_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=OLD_ERROR_HANDLING_MSG, category=DeprecationWarning)
        yield


@pytest.mark.usefixtures("no_spglib_deprecation_warning")
def test_call_spglib_success():
    before = _old_error_handling()
    dataset = call_spglib(spglib.get_symmetry_dataset, NACL_CELL, symprec=1e-3)
    assert dataset is not None
    assert dataset.number == 225
    # the global spglib setting is left as it was
    assert _old_error_handling() == before


@pytest.mark.usefixtures("no_spglib_deprecation_warning")
def test_call_spglib_failure_returns_none():
    before = _old_error_handling()
    assert call_spglib(spglib.get_symmetry_dataset, OVERLAPPING_CELL, symprec=1e-3) is None
    assert _old_error_handling() == before


def test_call_spglib_restores_flag_on_unexpected_error():
    if _old_error_handling() is None:
        pytest.skip("spglib has no OLD_ERROR_HANDLING switch")
    before = _old_error_handling()

    def boom():
        raise RuntimeError("unexpected")

    with pytest.raises(RuntimeError, match="unexpected"):
        call_spglib(boom)
    assert _old_error_handling() == before


@pytest.mark.parametrize("error_module", [None, types.SimpleNamespace()], ids=["no-error-module", "no-switch"])
def test_call_spglib_without_switch(monkeypatch, error_module):
    """spglib < 2.7 has no OLD_ERROR_HANDLING switch; spglib >= 3.0 removes it and always raises."""
    if error_module is None:
        monkeypatch.delattr(spglib, "error", raising=False)
    else:
        monkeypatch.setattr(spglib, "error", error_module)

    assert call_spglib(lambda value: value * 2, 21) == 42
    assert call_spglib(lambda: None) is None

    def raises_spglib_error():
        raise SpglibError("spglib failed")

    assert call_spglib(raises_spglib_error) is None


@pytest.mark.usefixtures("no_spglib_deprecation_warning")
def test_spacegroup_analyzer_emits_no_spglib_deprecation_warning():
    _, frac_coords, numbers = NACL_CELL
    structure = Structure(Lattice.cubic(5.64), numbers, frac_coords)
    sga = SpacegroupAnalyzer(structure)
    assert sga.get_space_group_number() == 225
    assert sga.get_refined_structure().composition.reduced_formula == "NaCl"
    assert sga.find_primitive() is not None
    assert len(sga.get_symmetry_operations()) == 192
    assert len(sga.get_ir_reciprocal_mesh((2, 2, 2))) > 0


def test_ignore_spglib_error_handling_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with ignore_spglib_error_handling_warning():
            warnings.warn(
                "Set OLD_ERROR_HANDLING to false and catch the errors directly.", DeprecationWarning, stacklevel=1
            )
            warnings.warn("unrelated", DeprecationWarning, stacklevel=1)
    assert [str(warning.message) for warning in caught] == ["unrelated"]
