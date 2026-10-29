from __future__ import annotations

import os
import shutil

import pytest
from pytest import approx

from pymatgen.core.structure import Molecule
from pymatgen.io.qchem.outputs import check_for_structure_changes
from pymatgen.io.xtb.outputs import CRESTOutput
from pymatgen.util.testing import TEST_FILES_DIR, MatSciTest

try:
    from openbabel import openbabel
except ImportError:
    openbabel = None
    print("OpenBabel not installed, parsed molecules structures will not be checked")

__author__ = "Alex Epstein"
__copyright__ = "Copyright 2020, The Materials Project"
__version__ = "0.1"

TEST_DIR = f"{TEST_FILES_DIR}/io/xtb/sample_CREST_output"
EXPECTED_DIR = f"{TEST_FILES_DIR}/io/xtb/expected_output"


class TestCRESTOutput(MatSciTest):
    """
    Checks that all attributes of CRESTOutput match the expected values for a
    sample CREST output directory.
    """

    def test_all(self):
        expected_cmd_options = {"g": "H2O", "c": "2"}
        expected_energies = [["-13.66580"] * 10, ["-13.66479"] * 27]
        expected_sorted_structures = [[], []]
        for filepath in os.listdir(EXPECTED_DIR):
            if filepath.endswith("xyz") and "_r" in filepath:
                n_conf = int(filepath.split("_")[0][-1])
                n_rot = int(filepath.split("_")[1].split(".")[0][-1])
                mol = Molecule.from_file(f"{EXPECTED_DIR}/{filepath}")
                expected_sorted_structures[n_conf].insert(n_rot, mol)

        crest_out = CRESTOutput(output_filename="crest_out.out", path=TEST_DIR)
        exp_best = Molecule.from_file(f"{EXPECTED_DIR}/expected_crest_best.xyz")
        for idx, c in enumerate(crest_out.sorted_structures_energies):
            for j, r in enumerate(c):
                if openbabel is not None:
                    assert check_for_structure_changes(r[0], expected_sorted_structures[idx][j]) == "no_change"
                assert float(r[1]) == approx(float(expected_energies[idx][j]), abs=1e-4)

        assert crest_out.properly_terminated
        if openbabel is not None:
            assert check_for_structure_changes(crest_out.lowest_energy_structure, exp_best) == "no_change"
        assert crest_out.cmd_options == expected_cmd_options


@pytest.mark.parametrize(
    ("cmd", "expected_options", "expected_charge"),
    [
        # trailing flag without a value used to raise IndexError
        ("crest crest_in.xyz -g H2O -c 2 --noreftopo", {"g": "H2O", "c": "2", "noreftopo": None}, 2),
        # double-dash flags were stored as "-chrg" so the charge was silently ignored
        ("crest crest_in.xyz --chrg 2 -g H2O", {"chrg": "2", "g": "H2O"}, 2),
        # negative charges were parsed as a flag ("1"), leaving "c" with value None (crash)
        ("crest crest_in.xyz -g H2O -c -1", {"g": "H2O", "c": "-1"}, -1),
        # multi-digit charges were truncated to their last digit
        ("crest crest_in.xyz -g H2O --chrg 12", {"g": "H2O", "chrg": "12"}, 12),
    ],
)
def test_crest_cmd_options(tmp_path, cmd, expected_options, expected_charge):
    out_dir = tmp_path / "crest"
    shutil.copytree(TEST_DIR, out_dir)
    out_file = out_dir / "crest_out.out"
    out_file.write_text(out_file.read_text().replace("crest crest_in.xyz -g H2O -c 2", cmd))

    crest_out = CRESTOutput(output_filename="crest_out.out", path=str(out_dir))
    assert crest_out.cmd_options == expected_options
    assert crest_out.lowest_energy_structure.charge == expected_charge


def test_crest_no_command(tmp_path):
    (tmp_path / "crest_out.out").write_text("no command here\n CREST terminated normally.\n")
    with pytest.raises(ValueError, match="No CREST command found"):
        CRESTOutput(output_filename="crest_out.out", path=str(tmp_path))
