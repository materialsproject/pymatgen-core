"""Tests for JDFTx input sets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pymatgen.io.jdftx.inputs import JDFTXInfile
from pymatgen.io.jdftx.sets import JdftxInputSet
from pymatgen.util.testing import TEST_FILES_DIR

from .inputs_test_utils import assert_identical_jif, ex_infile1_fname

ex_files_dir = Path(TEST_FILES_DIR) / "io" / "jdftx" / "example_files"


def test_jdftxinputset_from_directory():
    input_set = JdftxInputSet.from_file(ex_infile1_fname)
    jdftx_inputfile = JDFTXInfile.from_file(ex_infile1_fname)
    assert_identical_jif(input_set.jdftxinput, jdftx_inputfile)


def test_jdftxinputset_write_file(tmp_path):
    jdftx_inputfile = JDFTXInfile.from_file(ex_infile1_fname)
    input_set = JdftxInputSet(jdftx_inputfile, jdftx_inputfile.structure)
    input_set.write_input(tmp_path, infile="test.in")
    written_file = tmp_path / "test.in"
    read_jdftx_inputfile = JDFTXInfile.from_file(written_file)
    assert written_file.exists()
    assert_identical_jif(read_jdftx_inputfile, jdftx_inputfile)


def _effective_lattice(jif: JDFTXInfile) -> np.ndarray:
    """Lattice vectors (columns, bohr) as JDFTx builds them: the lattice tag scaled by latt-scale."""
    r = np.array([[jif["lattice"][f"R{i}{j}"] for j in range(3)] for i in range(3)])
    if "latt-scale" in jif:
        r = r * np.array([jif["latt-scale"][f"s{k}"] for k in range(3)])
    return r


@pytest.mark.parametrize(
    "text",
    [
        # Values with JDFTx's full printed precision (e.g. copied from an out file)
        "lattice 10 2 1 0 9 3 0 0 8\nion H 0.123456789012345 0.2 0.3 v 0.000012345678901 0.02 0.03 1\n",
        # A supercell made with latt-scale
        "lattice 10 2 1 0 9 3 0 0 8\nlatt-scale 2 2 1\nion H 0.1 0.2 0.3 1\n",
        "lattice 10 2 1 0 9 3 0 0 8\ncoords-type Cartesian\nion H 1.123456789012345 2 3 1\n",
    ],
)
def test_jdftxinputset_keeps_structure(tmp_path: Path, text: str):
    # Writing an input set rewrites its structure tags rather than adding a second copy of them
    (tmp_path / "in").write_text(text)
    original = JDFTXInfile.from_file(tmp_path / "in")
    JdftxInputSet.from_file(tmp_path / "in").write_input(tmp_path / "out", infile="init.in")
    written = JDFTXInfile.from_file(tmp_path / "out" / "init.in")
    assert len(written["ion"]) == len(original["ion"])
    np.testing.assert_allclose(_effective_lattice(written), _effective_lattice(original), rtol=1e-14)


def test_jdftxinputset_write_file_with_fixed_ions(tmp_path):
    """Ions with moveScale 0 must survive a read -> write round trip through JdftxInputSet."""
    infile = Path(TEST_FILES_DIR) / "io" / "jdftx" / "test_jdftx_in_files" / "ct_slab_001.in"
    jdftx_inputfile = JDFTXInfile.from_file(infile)
    input_set = JdftxInputSet.from_file(infile)
    input_set.write_input(tmp_path, infile="test.in")
    read_jdftx_inputfile = JDFTXInfile.from_file(tmp_path / "test.in")
    move_scales = [ion["moveScale"] for ion in jdftx_inputfile["ion"]]
    assert 0 in move_scales
    assert [ion["moveScale"] for ion in read_jdftx_inputfile["ion"]] == move_scales
