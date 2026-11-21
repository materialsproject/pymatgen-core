"""Tests for the JOutStructures class."""

from __future__ import annotations

import pytest
from pytest import approx

from pymatgen.io.jdftx.joutstructures import JOutStructure, JOutStructures, _get_joutstructure_step_bounds

from .outputs_test_utils import ex_outfileslice1 as ex_outslice1
from .outputs_test_utils import ex_outfileslice1_known as ex_outslice1_known


@pytest.mark.parametrize(
    ("ex_slice", "ex_slice_known", "opt_type"),
    [(ex_outslice1, ex_outslice1_known, "lattice")],
)
def test_jstructures(ex_slice: list[str], ex_slice_known: dict[str, float], opt_type: str):
    """General testing of the JOutStructures class.

    Args:
        ex_slice: The example slice to test.
        ex_slice_known: The known values of the example slice.
        opt_type: The optimization type
    """
    jstruct = JOutStructures._from_out_slice(ex_slice, opt_type=opt_type)
    assert isinstance(jstruct, JOutStructures)
    # Testing indexing
    assert isinstance(jstruct[0], JOutStructure)
    # Indexing should be equivalent to indexing the slices attribute
    assert jstruct[0] is jstruct.slices[0]
    # Testing initialization matches known values stored in ex_slice_known
    assert jstruct[0].elecmindata[0].mu == approx(ex_slice_known["mu0_0"])
    assert jstruct[0].elecmindata[-1].mu == approx(ex_slice_known["mu0_-1"])
    assert jstruct[-1].elecmindata[0].mu == approx(ex_slice_known["mu-1_0"])
    assert jstruct[-1].elecmindata[-1].mu == approx(ex_slice_known["mu-1_-1"])
    assert jstruct.elecmindata[-1].mu == approx(ex_slice_known["mu-1_-1"])
    assert jstruct.elecmindata.mu == approx(ex_slice_known["mu-1_-1"])
    assert jstruct.mu == approx(ex_slice_known["mu-1_-1"])
    assert jstruct["mu"] == approx(ex_slice_known["mu-1_-1"])
    assert jstruct[0].elecmindata[0].nelectrons == approx(ex_slice_known["nelec0_0"])
    assert jstruct[0].elecmindata[-1].nelectrons == approx(ex_slice_known["nelec0_-1"])
    assert jstruct[-1].elecmindata[0].nelectrons == approx(ex_slice_known["nelec-1_0"])
    assert jstruct[-1].elecmindata[-1].nelectrons == approx(ex_slice_known["nelec-1_-1"])
    assert jstruct.elecmindata[-1].nelectrons == approx(ex_slice_known["nelec-1_-1"])
    assert len(jstruct[0].elecmindata) == ex_slice_known["nEminSteps0"]
    assert len(jstruct[-1].elecmindata) == ex_slice_known["nEminSteps-1"]
    assert len(jstruct.elecmindata) == ex_slice_known["nEminSteps-1"]
    assert jstruct[0].etype == ex_slice_known["etype0"]
    assert approx(ex_slice_known["E0"]) == jstruct[0].e
    assert jstruct[-1].etype == ex_slice_known["etype-1"]
    assert jstruct.etype == ex_slice_known["etype-1"]
    assert approx(ex_slice_known["E-1"]) == jstruct[-1].e
    assert jstruct[0].elecmindata.converged == ex_slice_known["conv0"]
    assert jstruct[-1].elecmindata.converged == ex_slice_known["conv-1"]
    assert jstruct.elecmindata.converged == ex_slice_known["conv-1"]
    assert len(jstruct) == ex_slice_known["nGeomSteps"]
    # Lazy testing inherited attribute
    assert isinstance(jstruct.selective_dynamics, list)


@pytest.mark.parametrize("lowdin", [True, False], ids=["lowdin", "no_lowdin"])
def test_step_bounds(lowdin: bool):
    """Each geometry step gets one set of bounds, with or without a Lowdin block ending it.

    Electronic minimizations at rejected line-search positions are left out.
    """
    header = "-------- Electronic minimization -----------"
    step_end = ["#--- Lowdin population analysis ---", "# oxidation-state H +0.000 +0.000", ""] if lowdin else []
    lines = [
        header,
        "ElecMinimize: Converged (|Delta F|<1.000000e-08 for 2 iters).",
        "IonicMinimize: Iter:   0  F: -1.0  |grad|_K:  4.787e-02  t[s]:      0.35",
        *step_end,
        header,
        "ElecMinimize: Converged (|Delta F|<1.000000e-08 for 2 iters).",
        "IonicMinimize: \tWolfe criterion not satisfied: alpha: 1  (E-E0)/|gdotd0|: -0.99  gdotd/gdotd0: 0.99",
        header,
        "ElecMinimize: Converged (|Delta F|<1.000000e-08 for 2 iters).",
        "IonicMinimize: Iter:   1  F: -1.1  |grad|_K:  3.907e-02  alpha:  3.000e+00  linmin: -1.000e+00  t[s]:  0.45",
        *step_end,
        "End date and time: Sat Oct 24 12:52:49 2026  (Duration: 0-0:00:00.64)",
    ]
    steps = [lines[start:end] for start, end in _get_joutstructure_step_bounds(lines)]
    assert [[line for line in step if line.startswith("IonicMinimize: Iter:")] for step in steps] == [
        [lines[2]],
        [lines[8 + len(step_end)]],
    ]
    assert not any("Wolfe" in line for step in steps for line in step)
    assert all(step[0] == header for step in steps)
