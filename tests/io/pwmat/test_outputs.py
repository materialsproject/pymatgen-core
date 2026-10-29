from __future__ import annotations

import gzip
import shutil
from typing import TYPE_CHECKING

import pytest
from monty.io import zopen
from pytest import approx

from pymatgen.io.pwmat.outputs import DosSpin, Movement, OutFermi, Report
from pymatgen.util.testing import TEST_FILES_DIR, MatSciTest

if TYPE_CHECKING:
    from pathlib import Path

TEST_DIR = f"{TEST_FILES_DIR}/io/pwmat"


class TestMovement(MatSciTest):
    def test_init_and_properties(self):
        filepath = f"{TEST_DIR}/MOVEMENT.lzma"
        movement = Movement(filepath)
        assert movement.n_ionic_steps == len(movement.chunk_sizes) == 1
        assert movement.n_ionic_steps == len(movement.chunk_starts)
        assert movement.n_ionic_steps == len(movement.atom_configs)
        assert "atom_config" in movement.ionic_steps[0]
        assert "e_tot" in movement.ionic_steps[0]
        assert "atom_forces" in movement.ionic_steps[0]
        assert "virial" in movement.ionic_steps[0]
        assert "atom_energies" in movement.ionic_steps[0]
        assert movement.e_tots == approx(-357677.2281)
        assert movement.atom_configs[0] == movement.ionic_steps[0]["atom_config"]
        # Atomic-Energy block is present in MOVEMENT, so per-atom energies must be parsed
        assert movement.e_atoms.shape == (1, 72)
        assert movement.e_atoms[0][0] == approx(166.1493442)
        assert movement.atom_forces.shape == (1, 72, 3)
        assert movement.virials.shape == (1, 3, 3)
        assert movement.ionic_steps[0]["e_tot"] == approx(-357677.2281)

    @staticmethod
    def _write_multi_step_movement(path: Path, n_steps: int) -> list[float]:
        with zopen(f"{TEST_DIR}/MOVEMENT.lzma", mode="rt", encoding="utf-8") as file:
            chunk = file.read()
        e_tots = [-357677.2281 + 1000 * idx for idx in range(n_steps)]
        path.write_text("".join(chunk.replace("-0.3576772281E+06", f"{e_tot:.10E}") for e_tot in e_tots))
        return e_tots

    def test_multiple_ionic_steps(self):
        # each ionic step must be its own dict (they used to all alias the last step)
        e_tots = self._write_multi_step_movement(self.tmp_path / "MOVEMENT", 3)
        movement = Movement(self.tmp_path / "MOVEMENT")
        assert movement.n_ionic_steps == 3
        assert movement.e_tots == approx(e_tots)
        assert len({id(step) for step in movement.ionic_steps}) == 3

    @pytest.mark.parametrize(
        ("skip", "offset", "expected_idxs"),
        [(2, None, [0, 2, 4]), (2, 1, [1, 3]), (None, 3, [3, 4]), (None, None, [0, 1, 2, 3, 4])],
    )
    def test_ionic_step_skip_offset(self, skip, offset, expected_idxs):
        e_tots = self._write_multi_step_movement(self.tmp_path / "MOVEMENT", 5)
        movement = Movement(self.tmp_path / "MOVEMENT", ionic_step_skip=skip, ionic_step_offset=offset)
        assert movement.e_tots == approx([e_tots[idx] for idx in expected_idxs])


class TestOutFermi(MatSciTest):
    def test_init_and_properties(self):
        filepath = f"{TEST_DIR}/OUT.FERMI.lzma"
        out_fermi = OutFermi(filepath)
        assert out_fermi.e_fermi == approx(-2.359)


class TestReport(MatSciTest):
    def test_init_and_properties(self):
        filepath = f"{TEST_DIR}/REPORT"
        report = Report(filepath)
        assert report.spin == 1
        assert report.n_kpoints == 1
        assert report.n_bands == 10
        assert report.eigenvalues.shape == (1, 1, 10)
        assert report.kpoints.shape == (1, 3)
        assert report.kpoints_weight.shape == (1,)
        assert report.hsps == {}

    def test_compressed(self):
        # REPORT used to be read with linecache, which cannot read compressed files
        gz_path = self.tmp_path / "REPORT.gz"
        with open(f"{TEST_DIR}/REPORT", "rb") as fin, gzip.open(gz_path, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        report = Report(gz_path)
        ref = Report(f"{TEST_DIR}/REPORT")
        assert report.n_bands == ref.n_bands == 10
        assert report.eigenvalues == approx(ref.eigenvalues)
        assert report.kpoints == approx(ref.kpoints)


class TestDosSpin(MatSciTest):
    def test_init_and_properties(self):
        filepath = f"{TEST_DIR}/DOS.spinup_projected"
        dos_spin = DosSpin(filepath)
        assert dos_spin.dos.shape == (20, 21)
        assert dos_spin.labels == [
            "Energy",
            "Total",
            "Cr-3S",
            "Cr-3Pz",
            "Cr-3Px",
            "Cr-3Py",
            "Cr-4S",
            "Cr-3Dz2",
            "Cr-3Dxz",
            "Cr-3Dyz",
            "Cr-3D(x^2-y^2)",
            "Cr-3Dxy",
            "I-4Dz2",
            "I-4Dxz",
            "I-4Dyz",
            "I-4D(x^2-y^2)",
            "I-4Dxy",
            "I-5S",
            "I-5Pz",
            "I-5Px",
            "I-5Py",
        ]

    def test_compressed(self):
        # labels used to be read with linecache, which silently returned [] for compressed files
        gz_path = self.tmp_path / "DOS.spinup_projected.gz"
        with open(f"{TEST_DIR}/DOS.spinup_projected", "rb") as fin, gzip.open(gz_path, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        dos_spin = DosSpin(gz_path)
        assert dos_spin.labels == DosSpin(f"{TEST_DIR}/DOS.spinup_projected").labels
        assert dos_spin.get_partial_dos("Cr-3S").shape == (20,)
