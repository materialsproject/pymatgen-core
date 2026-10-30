from __future__ import annotations

import os
from shutil import which

import pytest
from monty.serialization import loadfn

from pymatgen.command_line import mcsqs_caller
from pymatgen.command_line.mcsqs_caller import run_mcsqs
from pymatgen.core.structure import Structure
from pymatgen.util.testing import TEST_FILES_DIR, MatSciTest

__author__ = "Handong Ling, Rachel Woods-Robinson"
__maintainer__ = "Handong Ling, Rachel Woods-Robinson"
__email__ = "handongling@berkeley.edu, rwoodsrobinson@lbl.gov"


TEST_DIR = f"{TEST_FILES_DIR}/io/atat/mcsqs"


class _FakePopen:
    """Stand-in for subprocess.Popen that fakes mcsqs output files in the cwd."""

    returncode = 0

    def __init__(self, cmd, *args, **kwargs):
        self.cmd = cmd

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def communicate(self, *args, **kwargs):
        for name in ("bestsqs.out", "bestcorr.out"):
            with open(name, mode="w", encoding="utf-8"):
                pass
        return b"", b""

    def kill(self):
        pass


@pytest.mark.parametrize("scaling", [2, 2.5])
def test_run_mcsqs_restores_cwd(tmp_path, monkeypatch, scaling):
    # run_mcsqs used to chdir into its run directory and only restore the cwd on timeout
    struct = MatSciTest.get_structure("Pb2TiZrO6").copy()
    struct.replace_species({"Ti": {"Ti": 0.5, "Zr": 0.5}, "Zr": {"Ti": 0.5, "Zr": 0.5}})
    monkeypatch.setattr(mcsqs_caller, "Popen", _FakePopen)
    monkeypatch.setattr(mcsqs_caller, "_parse_sqs_path", lambda path: "parsed")
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # run_mcsqs.__wrapped__ bypasses the @requires check for the mcsqs/str2cif binaries
    if scaling % 1 == 0:
        assert run_mcsqs.__wrapped__(struct, {2: 6}, scaling, directory=str(run_dir), instances=1) == "parsed"
    else:
        with pytest.raises(ValueError, match="should be an integer"):
            run_mcsqs.__wrapped__(struct, {2: 6}, scaling, directory=str(run_dir), instances=1)
    assert os.getcwd() == str(tmp_path)


@pytest.mark.skipif(not (which("mcsqs") and which("str2cif")), reason="mcsqs executable not present")
class TestMcsqsCaller(MatSciTest):
    def setup_method(self):
        self.pzt_structs = loadfn(f"{TEST_DIR}/pzt-structs.json")
        self.pzt_structs2 = loadfn(f"{TEST_DIR}/pzt-structs-2.json")
        self.struct = self.get_structure("Pb2TiZrO6")
        self.perfect_match_zzn_rs = loadfn(f"{TEST_DIR}/perfect_match_zzn_rs.json")

    def test_mcsqs_caller_supercell(self):
        struct = self.struct.copy()
        struct.replace_species({"Ti": {"Ti": 0.5, "Zr": 0.5}, "Zr": {"Ti": 0.5, "Zr": 0.5}})
        sqs = run_mcsqs(struct, {2: 6, 3: 4}, scaling=[2, 1, 1], search_time=0.01, instances=1)

        matches = [sqs.bestsqs.matches(s) for s in self.pzt_structs]
        assert any(matches)

        assert isinstance(sqs.bestsqs, Structure)

        # ensures specific keys are present in cluster parsing for use in atomate
        assert set(sqs.clusters[0]) == {
            "multiplicity",
            "coordinates",
            "longest_pair_length",
            "num_points_in_cluster",
        }
        assert set(sqs.clusters[0]["coordinates"][0]) == {
            "cluster_function",
            "coordinates",
            "num_possible_species",
        }

    def test_mcsqs_caller_total_atoms(self):
        struct = self.struct.copy()
        struct.replace_species({"Ti": {"Ti": 0.5, "Zr": 0.5}, "Zr": {"Ti": 0.5, "Zr": 0.5}})
        sqs = run_mcsqs(struct, {2: 6, 3: 4}, scaling=2, search_time=0.01, instances=1)

        matches = [sqs.bestsqs.matches(s) for s in self.pzt_structs2]
        assert any(matches)

    def test_mcsqs_caller_total_atoms_auto_instances(self):
        struct = self.struct.copy()
        struct.replace_species({"Ti": {"Ti": 0.5, "Zr": 0.5}, "Zr": {"Ti": 0.5, "Zr": 0.5}})
        sqs = run_mcsqs(struct, {2: 6, 3: 4}, scaling=2, search_time=0.01, instances=None)

        matches = [sqs.bestsqs.matches(s) for s in self.pzt_structs2]
        assert any(matches)

    def test_mcsqs_caller_parallel(self):
        # explicitly test with four instances

        struct = self.struct.copy()
        struct.replace_species({"Ti": {"Ti": 0.5, "Zr": 0.5}, "Zr": {"Ti": 0.5, "Zr": 0.5}})
        sqs = run_mcsqs(struct, {2: 6, 3: 4}, scaling=2, search_time=0.01, instances=4)

        matches = [sqs.bestsqs.matches(s) for s in self.pzt_structs2]
        assert any(matches)

    def test_mcsqs_perfect_match_error(self):
        scale = 32 / len(self.perfect_match_zzn_rs)
        sqs = run_mcsqs(
            self.perfect_match_zzn_rs,
            {2: 6, 3: 4},
            scaling=scale,
            search_time=1,
            instances=1,
        )

        assert sqs.objective_function == "Perfect_match"

    def test_mcsqs_perfect_match_error_parallel(self):
        scale = 32 / len(self.perfect_match_zzn_rs)
        sqs = run_mcsqs(
            self.perfect_match_zzn_rs,
            {2: 6, 3: 4},
            scaling=scale,
            search_time=1,
            instances=4,
        )

        assert sqs.objective_function == "Perfect_match"

    @pytest.mark.xfail(reason="Fix later #4455")
    def test_mcsqs_caller_runtime_error(self):
        struct = self.struct.copy()
        struct.replace_species({"Ti": {"Ti": 0.5, "Zr": 0.5}, "Zr": {"Ti": 0.5, "Zr": 0.5}})
        struct.replace_species({"Pb": {"Ti": 0.2, "Pb": 0.8}})
        struct.replace_species({"O": {"F": 0.8, "O": 0.2}})
        with pytest.raises(RuntimeError, match="mcsqs exited before timeout reached"):
            run_mcsqs(struct, {2: 6, 3: 4}, 10, search_time=0.000001)
