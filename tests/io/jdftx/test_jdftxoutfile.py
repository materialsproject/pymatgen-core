from __future__ import annotations

import re
import warnings
from itertools import combinations
from math import ceil
from time import process_time
from timeit import Timer
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pymatgen.core.units import Ha_to_eV, bohr_to_ang
from pymatgen.io.jdftx._output_utils import read_outfile_slices
from pymatgen.io.jdftx.generic_tags import JDFTXOptionWarning
from pymatgen.io.jdftx.inputs import JDFTXInfile
from pymatgen.io.jdftx.jdftxoutfileslice import JDFTXOutfileSlice
from pymatgen.io.jdftx.outputs import JDFTXOutfile, _jof_atr_from_last_slice

from .inputs_test_utils import aimd_infile_fname
from .outputs_test_utils import (
    debug_outfile_path,
    eopt_elecmin_outfile_path,
    eopt_scf_ionmin_outfile_path,
    eopt_scf_outfile_path,
    etot_etype_outfile_known_simple,
    etot_etype_outfile_path,
    ex_out_files_dir,
    example_aimd_outfile_known,
    example_aimd_outfile_known_site_properties,
    example_aimd_outfile_path,
    example_igp_aimd_outfile_known,
    example_igp_aimd_outfile_known_site_properties,
    example_igp_aimd_outfile_path,
    example_ionmin_outfile_known,
    example_ionmin_outfile_known_simple,
    example_ionmin_outfile_path,
    example_latmin_outfile_known,
    example_latmin_outfile_known_simple,
    example_latmin_outfile_path,
    example_sp_outfile_known,
    example_sp_outfile_known_simple,
    example_sp_outfile_path,
    example_vib_modes_known,
    example_vib_nrg_components,
    example_vib_outfile_path,
    fluid_site_params_outfile_path,
    hse06_outfile_path,
    jdftxoutfile_fromfile_matches_known,
    jdftxoutfile_fromfile_matches_known_simple,
    jdftxoutfile_matches_known,
    jdftxoutfile_matches_known_simple,
    noeigstats_outfile_known_simple,
    noeigstats_outfile_path,
    nolowdin_ionmin_outfile_path,
    nolowdin_latmin_outfile_path,
    nolowdin_md_outfile_path,
    o2_vector_spin_calc_dir,
    partial_lattice_init_outfile_known_lattice,
    partial_lattice_init_outfile_path,
    pcm_params_outfile_path,
    problem2_outfile_known_simple,
    problem2_outfile_path,
    problem3_outfile_path,
    rho_external_outfile_path,
    species_names_outfile_path,
    vdw_d2_outfile_path,
    vexternal_spin_outfile_path,
    vib_central_diff_outfile_path,
)
from .shared_test_utils import assert_same_value

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("filename", "known"),
    [
        (example_sp_outfile_path, example_sp_outfile_known),
        (example_latmin_outfile_path, example_latmin_outfile_known),
        (example_ionmin_outfile_path, example_ionmin_outfile_known),
    ],
)
def test_JDFTXOutfile_fromfile(filename: Path, known: dict):
    jdftxoutfile_fromfile_matches_known(filename, known)


# Step of the coarsest CPU clock the tests run on (time.process_time advances by 1/64 s on Windows)
CPU_CLOCK_STEP = 1 / 64


def _parse_time(filename: Path, skim_levels: tuple[str, ...] | None = None, number: int = 1) -> float:
    """Best-of-n CPU time of parsing an out file `number` times.

    CPU time leaves out time spent waiting on other processes (e.g. other pytest-xdist workers), and taking the
    fastest of several parses discards the ones slowed by interruptions, since noise can only add time.
    """
    timer = Timer(lambda: JDFTXOutfile.from_file(filename, skim_levels=skim_levels), timer=process_time)
    return min(timer.repeat(repeat=5, number=number))


@pytest.mark.parametrize(
    ("filename", "known", "known_simple"),
    [
        (example_sp_outfile_path, example_sp_outfile_known, example_sp_outfile_known_simple),
        (example_latmin_outfile_path, example_latmin_outfile_known, example_latmin_outfile_known_simple),
        (example_ionmin_outfile_path, example_ionmin_outfile_known, example_ionmin_outfile_known_simple),
    ],
)
def test_JDFTXOutfile_skimming(filename: Path, known: dict, known_simple: dict):
    jof_full = JDFTXOutfile.from_file(filename)
    # Count how many fewer objects are read at each depth level
    nless_outslices = len(jof_full.slices) - 1
    nless_last_geom = len(jof_full.slices[-1].jstrucs.slices) - 1
    nless_last_elec = len(jof_full.slices[-1].jstrucs.slices[-1].elecmindata.slices) - 1
    # Each CPU time can be off by one step of the clock, so a speedup is only tested where it is at least two steps:
    # - Only the skims that leave out most of the file are timed: whole out file slices if there are several, otherwise
    #   geometry steps. Skimming only the geometry steps of a file with several slices saves under half of the time,
    #   and skimming only electronic steps much less
    # - Each timing is of enough parses for the full parse to take 6 steps (one parse of these files can take under 1)
    timed_level = "outfile" if nless_outslices > 0 else "geom"
    time_full = number = 0
    if nless_outslices + nless_last_geom > 0:
        single_parse_time = min(Timer(lambda: JDFTXOutfile.from_file(filename)).repeat(repeat=3, number=1))
        number = ceil(6 * CPU_CLOCK_STEP / single_parse_time)
        time_full = _parse_time(filename, number=number)
    skim_level_options = ["outfile", "geom", "elec"]
    # Don't run outfile depth skimming test if only one slice
    if not nless_outslices > 0:
        skim_level_options.remove("outfile")
    # Generate all combinations of skim level depths to test
    skim_levelss: list[tuple[str, ...]] = []
    for r in range(1, len(skim_level_options) + 1):
        skim_levelss.extend(combinations(skim_level_options, r))
    levels_time_tested = 0
    time_save_logs = []

    for skim_levels in skim_levelss:
        jof_skim = JDFTXOutfile.from_file(filename, skim_levels=skim_levels)
        # Check that the skimmed version matches the full version where it should
        jdftxoutfile_matches_known(jof_skim, known)
        jdftxoutfile_matches_known_simple(jof_skim, known_simple)
        # Make sure skim levels are correctly applied
        if "outfile" in skim_levels:
            assert len(jof_skim.slices) == 1
        if "geom" in skim_levels:
            for oslice in jof_skim.slices:
                assert len(oslice.jstrucs.slices) == 1
        if "elec" in skim_levels:
            for oslice in jof_skim.slices:
                for gslice in oslice.jstrucs.slices:
                    assert len(gslice.elecmindata.slices) == 1
        # Make sure less objects are being read before testing speedup
        less_outfile_slices_read = nless_outslices if "outfile" in skim_levels else 0
        less_geom_slices_read = nless_last_geom if "geom" in skim_levels else 0
        less_elec_slices_read = nless_last_elec if "elec" in skim_levels else 0
        if timed_level in skim_levels and less_outfile_slices_read + less_geom_slices_read > 0:
            skim_time = _parse_time(filename, skim_levels, number=number)
            time_save_info = [
                filename.name,
                skim_levels,
                time_full,
                skim_time,
                less_outfile_slices_read,
                less_geom_slices_read,
                less_elec_slices_read,
            ]
            assert skim_time < time_full, time_save_info
            levels_time_tested += 1
            time_save_logs.append(time_save_info)
    if nless_outslices + nless_last_geom > 0:
        assert levels_time_tested > 0, "No skim levels tested - likely an error in test code."
    print(time_save_logs)


@pytest.mark.parametrize(
    ("filename", "known"),
    [
        (example_sp_outfile_path, example_sp_outfile_known_simple),
        (example_latmin_outfile_path, example_latmin_outfile_known_simple),
        (example_ionmin_outfile_path, example_ionmin_outfile_known_simple),
        (noeigstats_outfile_path, noeigstats_outfile_known_simple),
        (problem2_outfile_path, problem2_outfile_known_simple),
        (etot_etype_outfile_path, etot_etype_outfile_known_simple),
    ],
)
def test_JDFTXOutfile_fromfile_simple(filename: Path, known: dict):
    jdftxoutfile_fromfile_matches_known_simple(filename, known)


@pytest.mark.parametrize(
    ("filename", "latknown"),
    [
        (partial_lattice_init_outfile_path, partial_lattice_init_outfile_known_lattice),
    ],
)
def test_JDFTXOutfile_default_struc_inheritance(filename: Path, latknown: dict):
    jof = JDFTXOutfile.from_file(filename)
    lattice = jof.lattice
    for i in range(3):
        for j in range(3):
            assert pytest.approx(lattice[i][j]) == latknown[f"{i}{j}"]


# Ensure that a None slice (one that failed to parse) that occurs after the others leaves
# the values of the last parsed slice
@pytest.mark.parametrize(("ex_outfile_path"), [(partial_lattice_init_outfile_path)])
def test_none_on_partial(ex_outfile_path: Path):
    texts = read_outfile_slices(str(ex_outfile_path))
    texts0 = texts[:-1]

    slices = [
        JDFTXOutfileSlice._from_out_slice(text, [], [], is_bgw=False, none_on_error=False)
        for i, text in enumerate(texts0)
    ]
    outfile1 = JDFTXOutfile(slices=slices)
    slices.append(None)
    outfile2 = JDFTXOutfile(slices=slices)
    assert isinstance(outfile2, JDFTXOutfile)
    for var in _jof_atr_from_last_slice:
        assert_same_value(
            getattr(outfile1, var),
            getattr(outfile2, var),
        )


@pytest.mark.parametrize(
    ("example_vib_outfile_path", "example_vib_modes_known", "example_vib_nrg_components"),
    [
        (example_vib_outfile_path, example_vib_modes_known, example_vib_nrg_components),
    ],
)
def test_vib_parse(
    example_vib_outfile_path: Path, example_vib_modes_known: list[dict], example_vib_nrg_components: dict
):
    """
    Test that the vibration modes are parsed correctly from the outfile.
    """
    jdftxoutfile = JDFTXOutfile.from_file(example_vib_outfile_path)
    assert_same_value(jdftxoutfile.slices[-1].vibrational_modes, example_vib_modes_known)
    assert_same_value(jdftxoutfile.slices[-1].vibrational_energy_components, example_vib_nrg_components)


@pytest.mark.parametrize(
    ("aimd_outfile_path", "aimd_outfile_known", "aimd_outfile_known_site_properties"),
    [
        (example_aimd_outfile_path, example_aimd_outfile_known, example_aimd_outfile_known_site_properties),
        (example_igp_aimd_outfile_path, example_igp_aimd_outfile_known, example_igp_aimd_outfile_known_site_properties),
    ],
)
def test_aimd_parse(aimd_outfile_path: Path, aimd_outfile_known: dict, aimd_outfile_known_site_properties: dict):
    """
    Test that the AIMD properties are parsed correctly from the outfile.
    """
    jdftxoutfile = JDFTXOutfile.from_file(aimd_outfile_path)
    assert jdftxoutfile.slices[-1].is_md
    for key, val in aimd_outfile_known.items():
        assert hasattr(jdftxoutfile.slices[-1].jstrucs, key)
        assert_same_value(getattr(jdftxoutfile.slices[-1].jstrucs, key), val)
        assert hasattr(jdftxoutfile.slices[-1], key)
        assert_same_value(getattr(jdftxoutfile.slices[-1], key), val)
        assert hasattr(jdftxoutfile, key)
        assert_same_value(getattr(jdftxoutfile, key), val)
    for sp, val in aimd_outfile_known_site_properties.items():
        assert sp in jdftxoutfile.structure.site_properties
        assert_same_value(jdftxoutfile.structure.site_properties[sp], val)


def test_aimd_velocity_units():
    # The kinetic energy from the parsed velocities (angstrom/fs) matches the kinetic energy JDFTx reports
    jdftxoutfile = JDFTXOutfile.from_file(example_aimd_outfile_path)
    structure = jdftxoutfile.structure
    masses = np.array([site.specie.atomic_mass for site in structure])  # amu
    velocities = np.array(structure.site_properties["velocities"])  # angstrom/fs
    amu_ang2_per_fs2_to_ev = 1.66053906892e-27 * 1e10 / 1.602176634e-19  # CODATA amu and e
    ke = 0.5 * np.sum(masses[:, None] * velocities**2) * amu_ang2_per_fs2_to_ev
    assert ke == pytest.approx(jdftxoutfile.ke, rel=1e-4)


@pytest.mark.parametrize(
    ("outfile_path", "infile_path", "outfile_known"),
    [
        (example_aimd_outfile_path, aimd_infile_fname, example_aimd_outfile_known),
    ],
)
def test_outfile_to_infile(outfile_path: Path, infile_path: Path, outfile_known: dict):
    """
    Test that the JDFTXOutfile.to_jdftxinfile() method works correctly.
    """
    jdftxoutfile: JDFTXOutfile = JDFTXOutfile.from_file(outfile_path)
    JDFTXInfile.from_file(infile_path)
    new_infile: JDFTXInfile = jdftxoutfile.to_jdftxinfile()
    old_struc = jdftxoutfile.structure
    new_struc = new_infile.to_pmg_structure(new_infile, fill_site_properties=True)
    old_coords = old_struc.cart_coords
    new_coords = new_struc.cart_coords
    assert_same_value(old_coords, new_coords)
    old_velocities = old_struc.site_properties.get("velocities", None)
    new_velocities = new_struc.site_properties.get("velocities", None)
    [ov / nv for ov, nv in zip(old_velocities, new_velocities, strict=False)]
    assert_same_value(old_velocities, new_velocities)
    old_thermostat_velocity = jdftxoutfile.slices[-1].jstrucs.thermostat_velocity
    new_thermostat_velocity = np.array(new_infile["thermostat-velocity"]["v"])
    assert_same_value(old_thermostat_velocity, new_thermostat_velocity)

    # TODO: Do the old TODO related to filling out default input tags so the robust comparison below
    # can be used instead of the above.
    # assert new_infile.is_comparable_to(jdftxinfile, exclude_tags=["ion", "thermostat-velocity", "dump"])


def test_etot_bug_handling():
    outfile = JDFTXOutfile.from_file(problem3_outfile_path)
    assert len(outfile.slices) == 1
    assert len(outfile.slices[-1].jstrucs.slices) == 7
    # The first ionic step in this outfile has the etot bug
    assert outfile.slices[-1].jstrucs.slices[0].etype == "F"
    assert "F" not in outfile.slices[-1].jstrucs.slices[0].ecomponents
    assert_same_value(outfile.slices[-1].jstrucs.slices[0].e, -3.018425965201787 * Ha_to_eV)


@pytest.mark.parametrize(
    ("outfile_path", "eopt_type", "converged_reason", "geom_converged"),
    [
        (eopt_elecmin_outfile_path, "ElecMinimize", "|Delta F|<1.000000e-08 for 2 iters", True),
        (eopt_scf_outfile_path, "SCF", "|Delta E|<1.000000e-08 for 2 iters", True),
        # Stopped after 2 ionic steps, each converged electronically
        (eopt_scf_ionmin_outfile_path, "SCF", "|Delta E|<1.000000e-08 for 2 iters", False),
    ],
    ids=["elecmin", "scf", "scf_ionmin"],
)
def test_eopt_types(outfile_path: Path, eopt_type: str, converged_reason: str, geom_converged: bool):
    """Electronic steps are read for each electronic optimizer (logged as "ElecMinimize: Iter:" or "SCF: Cycle:")."""
    outfile = JDFTXOutfile.from_file(outfile_path, none_slice_on_error=False)
    jstrucs = outfile.slices[-1].jstrucs
    assert outfile.eopt_type == eopt_type
    assert outfile.converged is geom_converged
    for jstruc in jstrucs:
        assert jstruc.elecmindata.opt_type == eopt_type
        assert jstruc.elecmindata.converged
        assert jstruc.elecmindata.converged_reason == converged_reason
    # Every step JDFTx printed is parsed, with its step number, energy label and energy
    printed = [
        (int(m["nstep"]), m["etype"], float(m["e"]))
        for m in re.finditer(
            rf"^{re.escape(eopt_type)}: (?:Iter|Cycle):\s+(?P<nstep>\d+)\s+(?P<etype>F|G|Etot): (?P<e>\S+)",
            outfile_path.read_text(),
            flags=re.MULTILINE,
        )
    ]
    parsed = [(step.nstep, step.etype, step.e / Ha_to_eV) for jstruc in jstrucs for step in jstruc.elecmindata]
    assert [p[:2] for p in parsed] == [p[:2] for p in printed]
    assert [p[2] for p in parsed] == pytest.approx([p[2] for p in printed], rel=0, abs=1e-12)
    # SCF steps have no line-minimization data
    last_step = jstrucs[-1].elecmindata[-1]
    assert (last_step.grad_k is None) is (eopt_type == "SCF")
    assert last_step.t_s is not None


def test_eopt_types_agree():
    # Both electronic optimizers converge the same calculation to the same free energy
    elecmin = JDFTXOutfile.from_file(eopt_elecmin_outfile_path, none_slice_on_error=False)
    scf = JDFTXOutfile.from_file(eopt_scf_outfile_path, none_slice_on_error=False)
    assert scf.e == pytest.approx(elecmin.e, rel=0, abs=1e-8 * Ha_to_eV)
    assert scf.elec_nstep > 0
    assert elecmin.elec_nstep > 0


@pytest.mark.parametrize(
    ("outfile_path", "geom_opt_type", "geom_converged"),
    [
        (nolowdin_ionmin_outfile_path, "IonicMinimize", True),
        (nolowdin_md_outfile_path, "IonicDynamics", False),
        (nolowdin_latmin_outfile_path, "LatticeMinimize", False),
    ],
    ids=["ionmin", "md", "latmin"],
)
def test_no_lowdin_steps(outfile_path: Path, geom_opt_type: str, geom_converged: bool):
    """Every geometry step is kept when JDFTx skips the Lowdin block (no pseudopotential has atomic orbitals)."""
    text = outfile_path.read_text()
    assert "Lowdin" not in text
    outfile = JDFTXOutfile.from_file(outfile_path, none_slice_on_error=False)
    jstrucs = outfile.slices[-1].jstrucs
    assert jstrucs.opt_type == geom_opt_type
    assert outfile.converged is geom_converged
    # One structure per printed geometry step, with its step number and energy (MD prints PE with 6 decimals)
    step_key = "Step" if geom_opt_type == "IonicDynamics" else "Iter"
    printed = [
        (int(m["nstep"]), float(m["e"]))
        for m in re.finditer(
            rf"^{geom_opt_type}: {step_key}:\s+(?P<nstep>\d+)\s+(?:Etot|PE):\s+(?P<e>\S+)", text, flags=re.MULTILINE
        )
    ]
    assert [jstruc.nstep for jstruc in jstrucs] == [p[0] for p in printed]
    assert [jstruc.e / Ha_to_eV for jstruc in jstrucs] == pytest.approx([p[1] for p in printed], rel=0, abs=1e-6)
    # Each structure has the positions printed for its step, and its own converged electronic minimization
    printed_posns = [
        np.array([line.split()[2:5] for line in block.splitlines()], dtype=float) * bohr_to_ang
        for block in re.findall(r"^# Ionic positions in cartesian coordinates:\n((?:ion .*\n)+)", text, re.MULTILINE)
    ]
    assert len(printed_posns) == len(jstrucs)
    for jstruc, posns in zip(jstrucs, printed_posns, strict=True):
        assert jstruc.cart_coords == pytest.approx(posns)
        assert len(jstruc.elecmindata)
        assert jstruc.elecmindata.converged


def test_cut_iteration_line(tmp_path: Path):
    """A run killed mid-line (no energy on its last iteration line) loses only that geometry step."""
    lines = nolowdin_ionmin_outfile_path.read_text().splitlines()
    i = max(i for i, line in enumerate(lines) if line.startswith("ElecMinimize: Iter:"))
    cut_path = tmp_path / "cut.out"
    cut_path.write_text("\n".join([*lines[:i], lines[i].split("Etot")[0], "", "appended text"]) + "\n")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        outfile = JDFTXOutfile.from_file(cut_path)
    assert len(outfile.slices[-1].jstrucs) == 6
    # The last step is often cut short in a running calculation, so it is skipped silently
    assert not any("Skipping geometry step" in str(w.message) for w in record)


def test_unparsable_middle_step_warns(tmp_path: Path):
    """A step before the last that fails to parse is skipped with a warning giving its index."""
    lines = nolowdin_ionmin_outfile_path.read_text().splitlines()
    iter_lines = [i for i, line in enumerate(lines) if line.startswith("ElecMinimize: Iter:")]
    i = iter_lines[len(iter_lines) // 2]
    lines[i] = lines[i].split("Etot")[0]
    (tmp_path / "jdftx.out").write_text("\n".join(lines) + "\n")
    with pytest.warns(UserWarning, match=r"Skipping geometry step \d+ of \d+"):
        outfile = JDFTXOutfile.from_file(tmp_path / "jdftx.out")
    assert len(outfile.slices[-1].jstrucs) == 6


def test_unparsable_slices(tmp_path: Path):
    """A slice that fails to parse is dropped with a warning, and a file in which no slice parses raises."""
    lines = eopt_elecmin_outfile_path.read_text().splitlines()
    i = max(i for i, line in enumerate(lines) if line.startswith("ElecMinimize: Iter:"))
    # A single point killed mid-line, then appended to: its only step fails
    cut = [*lines[:i], lines[i].split(" F:")[0], "", "appended text"]
    cut_path = tmp_path / "cut.out"
    cut_path.write_text("\n".join(cut))
    with pytest.warns(UserWarning, match="Skipping an out file slice"), pytest.raises(ValueError, match="could be"):
        JDFTXOutfile.from_file(cut_path)
    # Appended to a finished run, only the cut run is dropped
    cut_path.write_text("\n".join(lines + cut))
    with pytest.warns(UserWarning, match="Skipping an out file slice"):
        assert len(JDFTXOutfile.from_file(cut_path).slices) == 1


@pytest.mark.parametrize("smearing", [True, False])
def test_unavailable_eigstats(tmp_path: Path, smearing: bool):
    # Without empty bands (e.g. an insulator with the default number of bands), JDFTx prints the LUMO, mu and gaps
    # as "unavailable"; the slice must still be parsed
    text = example_sp_outfile_path.read_text()
    for key in ["LUMO", "mu  ", "HOMO-LUMO gap", "Optical gap  "]:
        text = re.sub(rf"^(\s*{re.escape(key)}): .*$", r"\1: unavailable", text, flags=re.MULTILINE)
    if not smearing:
        text = re.sub(r"^elec-smearing .*\n", "", text, flags=re.MULTILINE)
    (tmp_path / "jdftx.out").write_text(text)
    outfile = JDFTXOutfile.from_file(tmp_path / "jdftx.out", none_slice_on_error=False)
    reference = JDFTXOutfile.from_file(example_sp_outfile_path)
    assert len(outfile.slices) == 1
    assert outfile.lumo is None
    assert outfile.egap is None
    assert outfile.homo == pytest.approx(reference.homo)
    # The Fermi level falls back to the mu printed during the electronic minimization
    assert outfile.efermi == pytest.approx(outfile.mu)
    if smearing:
        # The lumo filling needs the (unavailable) lumo energy, so metallicity is unknown
        assert outfile.homo_filling == pytest.approx(reference.homo_filling)
        assert outfile.lumo_filling is None
        assert outfile.is_metal is None
    else:
        # Without smearing the fillings don't depend on the band energies
        assert outfile.lumo_filling == 0
        assert outfile.is_metal is False
        # The edited echo implies Etot, but the minimization lines (from a smearing run) print F, which wins
        assert outfile.etype == "F"


def _printed_final_ions(text: str) -> tuple[str, list[list[str]]] | None:
    """Coordinate system and ion lines of the last positions block JDFTx printed (None if there isn't one)."""
    if "# Ionic positions in " not in text:
        return None
    header, *lines = text.rsplit("# Ionic positions in ", maxsplit=1)[-1].splitlines()
    return header.split()[0], [line.split() for line in lines if line.startswith("ion ")]


def _check_restart_matches_out_file(text: str, restart: JDFTXInfile, infile: JDFTXInfile, lattice_changed: bool):
    printed_ions = _printed_final_ions(text)
    if printed_ions is None:  # No positions were printed, so the input's ions are kept
        assert restart["ion"] == infile["ion"]
        return
    # The ions are exactly as JDFTx printed them, as if the final positions block was copied into a new input
    coords_type, printed = printed_ions
    assert restart.get("coords-type", "Lattice").lower() == coords_type.lower()
    assert len(restart["ion"]) == len(printed)
    for ion, row in zip(restart["ion"], printed, strict=True):
        assert ion["species-id"] == row[1]
        assert [ion[f"x{k}"] for k in range(3)] == [float(x) for x in row[2:5]]
        i_move_scale = 5
        if "v" in row:
            i = row.index("v")
            assert [ion["v"][f"vx{k}"] for k in range(3)] == [float(x) for x in row[i + 1 : i + 4]]
            i_move_scale = i + 4
        assert ion["moveScale"] == float(row[i_move_scale])
    # An unchanged lattice keeps the input's own lattice and latt-scale tags
    if not lattice_changed:
        assert restart["lattice"] == infile["lattice"]
        assert restart.get("latt-scale") == infile.get("latt-scale")


@pytest.mark.parametrize("outfile_path", sorted(ex_out_files_dir.glob("*.out")), ids=lambda path: path.name)
def test_to_jdftxinfile_matches_out_file(outfile_path: Path):
    outfile = JDFTXOutfile.from_file(outfile_path)
    restart = JDFTXInfile.from_str(str(outfile.to_jdftxinfile()))
    infile = outfile.slices[-1].infile
    lattice_changed = not np.array_equal(infile.structure.lattice.matrix, outfile.structure.lattice.matrix)
    _check_restart_matches_out_file(outfile_path.read_text(), restart, infile, lattice_changed)


def test_to_jdftxinfile_keeps_fractional_move_scale(tmp_path: Path):
    # A moveScale below 1 must not become 0 (a fixed ion) in the restart input
    text = re.sub(r"^(ion \S+(?: +\S+){3}) 1$", r"\1 0.5", example_ionmin_outfile_path.read_text(), flags=re.MULTILINE)
    (tmp_path / "jdftx.out").write_text(text)
    outfile = JDFTXOutfile.from_file(tmp_path / "jdftx.out")
    # The fixed ions (moveScale 0) are left as they are
    assert {ion["moveScale"] for ion in outfile.slices[-1].infile["ion"]} == {0.0, 0.5}
    restart = JDFTXInfile.from_str(str(outfile.to_jdftxinfile()))
    assert {ion["moveScale"] for ion in restart["ion"]} == {0.0, 0.5}
    _check_restart_matches_out_file(text, restart, outfile.slices[-1].infile, lattice_changed=False)


def test_to_jdftxinfile_thermostat_chain_length(tmp_path: Path):
    """A restart keeps every Nose-Hoover chain thermostat velocity, however many there are (chainLengthT)."""
    text = example_aimd_outfile_path.read_text().replace("chainLengthT 3", "chainLengthT 2")
    text = re.sub(r"^(thermostat-velocity \S+ \S+) \S+$", r"\1", text, flags=re.MULTILINE)
    (tmp_path / "jdftx.out").write_text(text)
    outfile = JDFTXOutfile.from_file(tmp_path / "jdftx.out")
    restart = JDFTXInfile.from_str(str(outfile.to_jdftxinfile()))
    assert restart["ionic-dynamics"]["chainLengthT"] == 2
    assert restart["thermostat-velocity"]["v"] == list(outfile.slices[-1].jstrucs.thermostat_velocity)
    assert len(restart["thermostat-velocity"]["v"]) == 2


def test_to_jdftxinfile_keeps_rho_external():
    # JDFTx always echoes rhoExternal with its yes/no includeSelfEnergy flag
    outfile = JDFTXOutfile.from_file(rho_external_outfile_path)
    expected = {"filename": "../n", "includeSelfEnergy": True}
    assert outfile.infile["rhoExternal"] == expected
    assert JDFTXInfile.from_str(str(outfile.to_jdftxinfile()))["rhoExternal"] == expected


def test_to_jdftxinfile_keeps_vexternal():
    outfile = JDFTXOutfile.from_file(vexternal_spin_outfile_path)
    expected = {"filenameUp": "../../rhoext/n", "filenameDn": "../../rhoext/n"}
    assert outfile.infile["Vexternal"] == expected
    assert JDFTXInfile.from_str(str(outfile.to_jdftxinfile()))["Vexternal"] == expected


def test_to_jdftxinfile_keeps_debug():
    outfile = JDFTXOutfile.from_file(debug_outfile_path)
    assert outfile.infile["debug"] == ["KpointsBasis", "Symmetries"]
    assert JDFTXInfile.from_str(str(outfile.to_jdftxinfile()))["debug"] == ["KpointsBasis", "Symmetries"]


def test_to_jdftxinfile_keeps_d2_dispersion():
    # JDFTx echoes DFT-D2 with the default scale as a bare "van-der-waals", which must not be dropped
    outfile = JDFTXOutfile.from_file(vdw_d2_outfile_path)
    assert outfile.infile["van-der-waals"] == 0.0
    restart = JDFTXInfile.from_str(str(outfile.to_jdftxinfile()))
    assert restart["van-der-waals"] == 0.0


def test_to_jdftxinfile_keeps_hse06():
    # pymatgen used to reject hyb-HSE06, so the restart silently fell back to JDFTx's default functional (gga-PBE).
    # Reading an out file does not repeat the warning given to whoever chose the functional.
    with warnings.catch_warnings():
        warnings.simplefilter("error", JDFTXOptionWarning)
        outfile = JDFTXOutfile.from_file(hse06_outfile_path)
        restart = outfile.to_jdftxinfile()
    assert outfile.infile["elec-ex-corr"] == "hyb-HSE06"
    assert outfile.xc_func == "hyb-HSE06"
    assert restart["elec-ex-corr"] == "hyb-HSE06"
    # Reading the restart as an input does warn
    with pytest.warns(JDFTXOptionWarning, match="hyb-gga-hse06"):
        assert JDFTXInfile.from_str(str(restart))["elec-ex-corr"] == "hyb-HSE06"


def test_to_jdftxinfile_leaves_out_nan():
    """JDFTx echoes unset ionic-dynamics P0/stress0 as nan and can't read it back, so it refused every MD restart."""
    outfile = JDFTXOutfile.from_file(nolowdin_md_outfile_path)
    restart = outfile.to_jdftxinfile()
    assert "nan" not in str(restart).split()
    assert "P0" not in restart["ionic-dynamics"]
    assert "stress0" not in restart["ionic-dynamics"]
    assert restart["ionic-dynamics"]["nSteps"] == 3
    # The parsed input keeps them
    assert np.isnan(outfile.infile["ionic-dynamics"]["P0"])


def test_to_jdftxinfile_keeps_vibrations():
    """JDFTx since 2026-08 echoes iConfiguration/nConfigurations, which pymatgen didn't know, so it dropped the line."""
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Skipping invalid tag line")
        outfile = JDFTXOutfile.from_file(vib_central_diff_outfile_path)
    expected = {
        "dr": 0.01,
        "centralDiff": True,
        "useConstraints": False,
        "translationSym": True,
        "rotationSym": False,
        "omegaMin": 0.0002,
        "T": 298.0,
        "omegaResolution": 0.0001,
        "dumpK": False,
        "iConfiguration": -1,
        "nConfigurations": -1,
    }
    assert outfile.infile["vibrations"] == expected
    assert JDFTXInfile.from_str(str(outfile.to_jdftxinfile()))["vibrations"] == expected


def test_species_names_from_out_file():
    """Species named after their pseudopotential files (H1 and H2) used to make the whole out file unreadable."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        outfile = JDFTXOutfile.from_file(species_names_outfile_path)
    assert len(outfile.slices) == 1
    steps = outfile.slices[-1].jstrucs
    assert len(steps) == 3
    for structure in [outfile.slices[-1].initial_structure, *[step.structure for step in steps], outfile.structure]:
        assert [site.specie.symbol for site in structure] == ["H", "H"]
        assert structure.site_properties["jdftx_species"] == ["H1", "H2"]
    # JDFTx prints Lowdin values by species name ("# oxidation-state H1 +0.015")
    np.testing.assert_allclose(steps[0].charges, [0.015, -0.005])
    np.testing.assert_allclose(steps[0].magnetic_moments, [0.538, -0.539])
    assert outfile.atom_elements == ["H", "H"]
    assert outfile.atom_elements_int == [1, 1]
    assert outfile.atom_types == ["H1", "H2"]
    # Electrons are counted per species, from each one's pseudopotential
    assert (outfile.total_electrons_uncharged, outfile.semicore_electrons_uncharged) == (2, 0)


def test_to_jdftxinfile_keeps_species_names():
    # Checked with JDFTx: it accepts the restart, with both species and the final positions it printed
    outfile = JDFTXOutfile.from_file(species_names_outfile_path)
    restart = outfile.to_jdftxinfile()
    assert [ion["species-id"] for ion in restart["ion"]] == ["H1", "H2"]
    assert restart["ion-species"] == ["SG15/H1.upf", "SG15/H2.upf"]
    assert restart["ion"][0]["x2"] == 1.085877149994234  # As printed, not converted to angstrom and back
    assert [group["species-id"] for group in restart["initial-magnetic-moments"]["species-values"]] == ["H1", "H2"]


def test_species_element_differing_from_pseudopotential_warns(tmp_path: Path):
    # UPF files state their element, which pymatgen otherwise takes from the species name
    lines = species_names_outfile_path.read_text().splitlines(keepends=True)
    i = next(i for i, line in enumerate(lines) if line.startswith("Reading pseudopotential file 'SG15/H2.upf'"))
    lines[i + 1] = lines[i + 1].replace("'H' pseudopotential", "'O' pseudopotential")
    (tmp_path / "jdftx.out").write_text("".join(lines))
    with pytest.warns(UserWarning, match=re.escape("other elements: H2 (read as H, file for O)")):
        JDFTXOutfile.from_file(tmp_path / "jdftx.out")


def test_to_jdftxinfile_keeps_fluid_site_params():
    # JDFTx's fluid-site-params echo used to be unreadable, so the restart silently ran with the default site parameters
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        outfile = JDFTXOutfile.from_file(fluid_site_params_outfile_path)
        restart = outfile.to_jdftxinfile()
    # JDFTx echoes every site, and a bare line for the second input line, which isn't stored
    entries = outfile.infile["fluid-site-params"]
    assert [(entry["component"], entry["siteName"]) for entry in entries] == [("H2O", "O"), ("H2O", "H")]
    assert entries[0]["sigmaNuc"] == 0.5
    assert entries[1]["aPol"] == 0.4
    # The restart leaves out the "Rhs 0" that JDFTx echoes for H but refuses as input
    lines = [line.split() for line in str(restart).splitlines() if line.startswith("fluid-site-params")]
    assert [line[:3] for line in lines] == [["fluid-site-params", "H2O", "O"], ["fluid-site-params", "H2O", "H"]]
    assert "Rhs" in lines[0]
    assert "Rhs" not in lines[1]


def test_to_jdftxinfile_keeps_classical_dft_fluid_solvent():
    # For ClassicalDFT, JDFTx echoes the "translation" key as "translationMode", which its input doesn't accept. The
    # line used to be skipped, so the restart silently ran JDFTx's default solvent (water with default parameters).
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        outfile = JDFTXOutfile.from_file(fluid_site_params_outfile_path)
        restart = outfile.to_jdftxinfile()
    solvent = outfile.infile["fluid-solvent"][0]
    assert (solvent["name"], solvent["translation"], solvent["epsLJ"], solvent["quad_nBeta"]) == (
        "H2O",
        "LinearSpline",
        0.0,
        0,
    )
    # The restart (checked with JDFTx: it echoes the same fluid-solvent) leaves out the zeros JDFTx refuses as input
    (line,) = [line.split() for line in str(restart).splitlines() if line.startswith("fluid-solvent")]
    assert line[:4] == ["fluid-solvent", "H2O", "55.338", "ScalarEOS"]
    assert "translation" in line
    assert not any(key in line for key in ("translationMode", "epsLJ", "quad_nBeta"))


def test_to_jdftxinfile_keeps_pcm_params():
    # JDFTx's pcm-params echo used to be unreadable, so the restart silently ran with the default PCM parameters
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        outfile = JDFTXOutfile.from_file(pcm_params_outfile_path)
    params = outfile.infile["pcm-params"]
    assert params["lMax"] == 3
    assert params["sigma"] == pytest.approx(0.7)
    assert params["cavityTension"] == pytest.approx(5e-06)
    # The restart is valid input (checked with JDFTx: it echoes the same pcm-params as the run did). It leaves out the
    # zeros JDFTx echoes for unset keys but refuses as input.
    unset = ["rhoMin", "rhoMax", "rhoDelta", "Ztot", "screenOverride"]
    restart = JDFTXInfile.from_str(str(outfile.to_jdftxinfile()))
    assert restart["pcm-params"] == {key: value for key, value in params.items() if key not in unset}


@pytest.mark.parametrize("latt_scale", ["2 2 1", "1.5 0.75 2"])
def test_to_jdftxinfile_with_latt_scale(tmp_path: Path, latt_scale: str):
    # A supercell made with latt-scale is not scaled a second time in the restart input
    text = example_aimd_outfile_path.read_text().replace("latt-scale 1 1 1 ", f"latt-scale {latt_scale} ", 1)
    (tmp_path / "jdftx.out").write_text(text)
    outfile = JDFTXOutfile.from_file(tmp_path / "jdftx.out")
    restart = JDFTXInfile.from_str(str(outfile.to_jdftxinfile()))
    _check_restart_matches_out_file(text, restart, outfile.slices[-1].infile, lattice_changed=False)


def test_forces_in_lattice_coordinates():
    """Forces printed in (covariant) lattice coordinates must be converted with inv(R^T) to Cartesian.

    noeigstats.out is a two-atom Si cell with a non-symmetric lattice matrix where the residual forces are printed in
    lattice coordinates. By symmetry, the Cartesian force on each atom must lie along the Si-Si bond.
    """
    jout = JDFTXOutfile.from_file(noeigstats_outfile_path)
    structure = jout.structure
    force_lattice = np.array(
        [
            [-0.001029086579454, -0.001017871569470, -0.001000410553100],
            [0.001050974414728, 0.001046953088279, 0.001046523989185],
        ]
    )
    expected = force_lattice @ np.linalg.inv(structure.lattice.matrix).T * Ha_to_eV
    assert jout.forces == pytest.approx(expected)
    bond = structure.cart_coords[0] - structure.cart_coords[1]
    cos_angle = np.dot(jout.forces[0], bond) / np.linalg.norm(jout.forces[0]) / np.linalg.norm(bond)
    assert cos_angle == pytest.approx(-1, abs=1e-3)


def test_trajectory_does_not_mutate_slices():
    """Building the combined trajectory must not extend the first slice's trajectory in place."""
    jout = JDFTXOutfile.from_file(example_latmin_outfile_path)
    assert len(jout.slices) > 1
    slice_lens = [len(slc.trajectory) for slc in jout.slices]
    assert len(jout.trajectory) == sum(slice_lens)
    assert slice_lens[0] == len(jout.slices[0].jstrucs)
    assert jout.trajectory is not jout.slices[0].trajectory


def test_as_dict_with_none_slice():
    """Slices that failed to parse are stored as None and must not break as_dict."""
    jout = JDFTXOutfile.from_file(example_sp_outfile_path)
    jout_with_none = JDFTXOutfile(slices=[None, *jout.slices])
    dct = jout_with_none.as_dict()
    assert dct["slices"][0] is None
    assert len(dct["slices"]) == 2


def test_as_dict_without_slices():
    """An out file without a readable slice gives an empty JDFTXOutfile, whose values are None."""
    jout = JDFTXOutfile(slices=[])
    dct = jout.as_dict()
    assert dct["slices"] == []
    assert dct["prefix"] is None
    assert jout.e is None


@pytest.mark.parametrize("skim_levels", [["geom"], ["geom", "elec"], ["outfile", "geom"]], ids="-".join)
@pytest.mark.parametrize(
    "outfile_path", [example_vib_outfile_path, vib_central_diff_outfile_path], ids=lambda path: path.name
)
def test_geom_skim_of_vibrations(outfile_path: Path, skim_levels: list[str]):
    """Vibrations runs print no geometry, so skimming to the last geometry step used to leave no slice at all."""
    full = JDFTXOutfile.from_file(outfile_path)
    skim = JDFTXOutfile.from_file(outfile_path, skim_levels=skim_levels, none_slice_on_error=False)
    assert len(skim.slices) == 1
    assert len(skim.slices[-1].jstrucs.slices) == 1
    assert skim.e == full.e
    assert skim.structure == full.structure
    assert_same_value(skim.vibrational_modes, full.vibrational_modes)


def test_geom_skim_of_run_stopped_in_first_step(tmp_path: Path):
    """A run stopped before its first geometry was printed skims to the step the full parse gives."""
    lines = nolowdin_ionmin_outfile_path.read_text().splitlines(keepends=True)
    first_step = next(i for i, line in enumerate(lines) if "-------- Electronic minimization" in line)
    first_geometry = next(i for i, line in enumerate(lines) if i > first_step and line.startswith("# Ionic positions"))
    (tmp_path / "jdftx.out").write_text("".join(lines[: (first_step + first_geometry) // 2]))
    full = JDFTXOutfile.from_file(tmp_path / "jdftx.out")
    skim = JDFTXOutfile.from_file(tmp_path / "jdftx.out", skim_levels=["geom"], none_slice_on_error=False)
    assert_same_value(skim.slices[-1].jstrucs[-1].as_dict(), full.slices[-1].jstrucs[-1].as_dict())


@pytest.mark.parametrize("spintype", ["vector-spin", "spin-orbit"])
def test_noncollinear_spin(spintype: str, tmp_path: Path):
    """Noncollinear out files crashed from_file. They have one set of (spinor) states, each band holds one electron,
    and vector-spin prints each Lowdin moment as |M|, theta and phi."""
    text = (o2_vector_spin_calc_dir / "out").read_text()
    if spintype == "spin-orbit":  # The same states; JDFTx prints no moments for spin-orbit
        text = re.sub(r"^spintype vector-spin$", "spintype spin-orbit", text, flags=re.MULTILINE)
        text = re.sub(r"^# magnetic-moments .*\n", "", text, flags=re.MULTILINE)
    (tmp_path / "jdftx.out").write_text(text)
    outfile = JDFTXOutfile.from_file(tmp_path / "jdftx.out", none_slice_on_error=False)
    assert outfile.nspin == 1
    assert outfile.homo_filling == pytest.approx(1.0)
    assert outfile.lumo_filling == pytest.approx(0.0)
    assert not outfile.is_metal
    assert JDFTXInfile.from_str(str(outfile.to_jdftxinfile()))["spintype"] == spintype
    if spintype == "vector-spin":
        # "# magnetic-moments O   0.996 29.9 45.0   0.997 29.8 45.4", as Cartesian vectors
        m, theta, phi = np.array([[0.996, 29.9, 45.0], [0.997, 29.8, 45.4]]).T
        theta, phi = np.radians(theta), np.radians(phi)
        expected = m[:, None] * np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]).T
        assert np.array(outfile.structure.site_properties["magmom"]) == pytest.approx(expected)


@pytest.mark.parametrize("skim_levels", [None, ["outfile", "geom"]])
def test_skip_props(skim_levels: list[str] | None):
    """Each skipped property is left unparsed and nothing else changes ("struct" used to leave no slice at all, and
    "charges" skipped nothing)."""
    full = JDFTXOutfile.from_file(species_names_outfile_path, skim_levels=skim_levels)
    skipped = {
        prop: JDFTXOutfile.from_file(species_names_outfile_path, skim_levels=skim_levels, skip_props=[prop])
        for prop in ("struct", "forces", "charges")
    }
    for outfile in skipped.values():
        assert len(outfile.slices[-1].jstrucs) == len(full.slices[-1].jstrucs)
        assert outfile.e == pytest.approx(full.e)
    # Without the position lines, each step keeps the input structure
    assert skipped["struct"].structure.cart_coords == pytest.approx(full.slices[-1].initial_structure.cart_coords)
    assert skipped["struct"].structure.cart_coords != pytest.approx(full.structure.cart_coords)
    assert skipped["forces"].forces is None
    assert full.forces is not None
    assert "charges" not in skipped["charges"].structure.site_properties
    assert "charges" in full.structure.site_properties
