from __future__ import annotations

import re
import shutil
import sys
import warnings

import pytest
from numpy.testing import assert_allclose
from pytest import approx

from pymatgen.command_line import bader_caller
from pymatgen.command_line.bader_caller import BaderAnalysis, bader_analysis_from_path
from pymatgen.io.vasp.outputs import Chgcar
from pymatgen.util.testing import TEST_FILES_DIR, VASP_IN_DIR, VASP_OUT_DIR, MatSciTest

TEST_DIR = f"{TEST_FILES_DIR}/command_line/bader"


@pytest.mark.parametrize("parse_atomic_densities", [True, False])
def test_parse_atomic_densities_deprecation(parse_atomic_densities):
    with (
        pytest.warns(DeprecationWarning, match="parse_atomic_densities is deprecated"),
        pytest.raises(ValueError, match="You must provide either a Cube file or a CHGCAR"),
    ):
        BaderAnalysis(bader_path="bader", parse_atomic_densities=parse_atomic_densities)


FAKE_ACF = """\
    #         X           Y           Z       CHARGE      MIN DIST    ATOMIC VOL
 --------------------------------------------------------------------------------
    1      0.0000      0.0000      0.0000      1.0000      1.0000      10.0000
 --------------------------------------------------------------------------------
    VACUUM CHARGE:               0.0000
    VACUUM VOLUME:               0.0000
    NUMBER OF ELECTRONS:         1.0000
"""


def _write_fake_bader(path, banner: str) -> str:
    """Write a fake bader executable that prints banner and writes ACF.dat."""
    exe = path / "fake_bader"
    exe.write_text(f"#!/bin/sh\necho '{banner}'\ncat > ACF.dat <<'EOF'\n{FAKE_ACF}EOF\n")
    exe.chmod(0o755)
    return str(exe)


@pytest.mark.skipif(sys.platform == "win32", reason="fake bader is a POSIX shell script")
@pytest.mark.parametrize(
    ("banner", "version"),
    [("  GRID BASED BADER ANALYSIS  (Version 1.05 08/19/23)", 1.05), ("unexpected output", -1)],
)
def test_relative_path_and_version(tmp_path, monkeypatch, banner, version):
    # relative paths used to be resolved inside ScratchDir (FileNotFoundError) and a
    # short/unexpected version banner raised IndexError
    bader_exe = _write_fake_bader(tmp_path, banner)
    shutil.copy(f"{TEST_DIR}/CHGCAR.gz", tmp_path / "CHGCAR.gz")
    monkeypatch.chdir(tmp_path)
    ctx = pytest.warns(UserWarning, match="outdated") if version < 1 else warnings.catch_warnings()
    with ctx:
        analysis = BaderAnalysis(chgcar_filename="CHGCAR.gz", bader_path=bader_exe)
    assert analysis.version == approx(version)
    assert analysis.data[0]["charge"] == approx(1)
    assert analysis.nelectrons == approx(1)


def test_from_path_missing_chgcar(tmp_path):
    # the "not found" check compared against None, but the helper returns "", so the
    # missing CHGCAR surfaced later as an unrelated error (e.g. bader not installed)
    with (
        pytest.warns(UserWarning, match="Could not find 'CHGCAR'"),
        pytest.raises(FileNotFoundError, match="Could not find CHGCAR"),
    ):
        BaderAnalysis.from_path(str(tmp_path))


def test_bader_analysis_from_objects_does_not_mutate_chgcar(monkeypatch):
    # the spin-polarized branch used to overwrite chgcar.data["total"] with the
    # magnetization density and set chgcar.is_spin_polarized = False on the input
    chgcar = Chgcar.from_file(f"{TEST_DIR}/CHGCAR.gz")
    chgcar.data["diff"] = 0.1 * chgcar.data["total"]
    chgcar.is_spin_polarized = True
    total_before = chgcar.data["total"].copy()
    written_mag = {}

    class FakeBaderAnalysis:
        def __init__(self, chgcar_filename, **kwargs):
            if chgcar_filename.endswith("CHGCAR_mag"):
                written_mag["total"] = Chgcar.from_file(chgcar_filename).data["total"]
            self.data = [{"min_dist": 1.0, "charge": 1.0, "atomic_vol": 10.0}]
            self.vacuum_charge = self.vacuum_volume = 0.0
            self.version = 1.05

    monkeypatch.setattr(bader_caller, "BaderAnalysis", FakeBaderAnalysis)
    summary = bader_caller.bader_analysis_from_objects(chgcar)

    assert summary["magmom"] == [1.0]
    assert chgcar.is_spin_polarized
    assert_allclose(chgcar.data["total"], total_before)
    assert_allclose(written_mag["total"], chgcar.data["diff"], rtol=1e-4)


@pytest.mark.skipif(not shutil.which("bader"), reason="bader executable not present")
class TestBaderAnalysis(MatSciTest):
    def setup_method(self):
        warnings.catch_warnings()

    def test_init(self):
        # test with reference file
        analysis = BaderAnalysis(
            chgcar_filename=f"{VASP_OUT_DIR}/CHGCAR.Fe3O4.gz",
            potcar_filename=f"{VASP_IN_DIR}/POTCAR_Fe3O4.gz",
            chgref_filename=f"{VASP_OUT_DIR}/CHGCAR.Fe3O4_ref.gz",
        )
        assert len(analysis.data) == 14
        assert analysis.data[0]["charge"] == approx(6.6136782, abs=1e-3)
        assert analysis.data[0]["charge"] == analysis.get_charge(0)
        assert analysis.nelectrons == 96
        assert analysis.vacuum_charge == approx(0)
        results = [
            -1.3863218,
            -1.3812175,
            -1.3812175,
            -1.2615902,
            -1.3812175,
            -1.3862971,
            1.021523,
            1.024357,
            1.021523,
            1.021523,
            1.021523,
            1.021523,
            1.021523,
            1.024357,
        ]
        for idx in range(14):
            assert results[idx] == approx(analysis.get_charge_transfer(idx), abs=1e-3)
        assert analysis.get_partial_charge(0) == -analysis.get_charge_transfer(0)
        struct = analysis.get_oxidation_state_decorated_structure()
        assert struct[0].specie.oxi_state == approx(1.3863218, abs=1e-3)

        # make sure bader still runs without reference file
        analysis = BaderAnalysis(chgcar_filename=f"{VASP_OUT_DIR}/CHGCAR.Fe3O4.gz")
        assert len(analysis.data) == 14

        # Test Cube file format parsing
        shutil.copytree(TEST_DIR, self.tmp_path, dirs_exist_ok=True)
        analysis = BaderAnalysis(cube_filename=f"{TEST_DIR}/elec.cube.gz")
        assert len(analysis.data) == 9

    def test_from_path(self):
        # we need to create two copies of input files since monty decompressing files
        # deletes the compressed version which can't happen twice in same directory
        shutil.copytree(TEST_DIR, direct_dir := f"{self.tmp_path}/direct", dirs_exist_ok=True)
        shutil.copytree(TEST_DIR, from_path_dir := f"{self.tmp_path}/from_path", dirs_exist_ok=True)
        chgcar_path = f"{direct_dir}/CHGCAR.gz"
        chgref_path = f"{direct_dir}/_CHGCAR_sum.gz"

        analysis = BaderAnalysis(chgcar_filename=chgcar_path, chgref_filename=chgref_path)
        analysis_from_path = BaderAnalysis.from_path(from_path_dir)

        for key, val_from_path in analysis_from_path.summary.items():
            val = analysis.summary[key]
            if isinstance(val_from_path, bool | str):
                assert val == val_from_path, f"{key=}"
            elif key == "charge":
                assert_allclose(val, val_from_path, atol=1e-5)

    def test_bader_analysis_from_path(self):
        """
        Reference summary dict (with bader 1.0)

        summary_ref = {
            "magmom": [4.298761, 4.221997, 4.221997, 3.816685, 4.221997, 4.298763, 0.36292, 0.370516, 0.36292,
                0.36292, 0.36292, 0.36292, 0.36292, 0.370516],
            "min_dist": [0.835789, 0.92947, 0.92947, 0.973007, 0.92947, 0.835789, 0.94067, 0.817381, 0.94067,
                0.94067, 0.94067, 0.94067, 0.94067, 0.817381],
            "vacuum_charge": 0.0,
            "vacuum_volume": 0.0,
            "atomic_volume": [9.922887, 8.175158, 8.175158, 9.265802, 8.175158, 9.923233, 12.382546, 12.566972,
                12.382546, 12.382546, 12.382546, 12.382546, 12.382546, 12.566972],
            "charge": [12.248132, 12.26177, 12.26177, 12.600596, 12.26177, 12.248143, 7.267303, 7.256998,
                7.267303, 7.267303, 7.267303, 7.267303, 7.267303, 7.256998],
            "bader_version": 1.0,
            "reference_used": True,
        }
        """

        summary = bader_analysis_from_path(TEST_DIR)

        assert set(summary) == {
            "magmom",
            "min_dist",
            "vacuum_charge",
            "vacuum_volume",
            "atomic_volume",
            "charge",
            "bader_version",
            "reference_used",
        }
        assert summary["reference_used"]
        assert sum(summary["magmom"]) == approx(28, abs=1e-1)

    def test_parse_atomic_densities_deprecation(self):
        with pytest.warns(DeprecationWarning, match="parse_atomic_densities is deprecated"):
            analysis = BaderAnalysis(
                chgcar_filename=f"{VASP_OUT_DIR}/CHGCAR.Fe3O4.gz",
                parse_atomic_densities=True,
            )

        assert not hasattr(analysis, "atomic_densities")
        assert "charge_densities" not in analysis.summary

    def test_missing_file_bader_exe_path(self):
        # Mock which("bader") to return None so we always fall back to use bader_exe_path
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setenv("PATH", "")

            with pytest.raises(
                RuntimeError,
                match=re.escape("Requires bader or bader.exe to be in the PATH or the absolute path"),
            ):
                BaderAnalysis(chgcar_filename=f"{VASP_OUT_DIR}/CHGCAR.Fe3O4.gz")
