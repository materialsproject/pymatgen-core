"""Tests for the output_utils module."""

from __future__ import annotations

import os
import re

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pymatgen.io.jdftx._output_utils import (
    _get_nbands_from_bandfile_filepath,
    _get_orb_label_list,
    find_first_range_key,
    get_proj_tju_from_file,
    get_start_lines,
    read_outfile_slices,
)
from pymatgen.io.jdftx.joutstructures import _get_joutstructures_start_idx
from pymatgen.io.jdftx.outputs import _find_jdftx_out_file

from .outputs_test_utils import n2_ex_calc_dir, noeigstats_outfile_path
from .shared_test_utils import write_mt_file


def test_get_start_lines():
    """Test the get_start_lines function.

    This function is used to find the start of the JDFTx calculations in the outfile.
    It tests the behavior to make sure the correct errors are raised on empty files and files without
    the pattern to find the start of the JDFTx calculations.
    """
    with pytest.raises(ValueError, match=re.escape("Outfile parser fed an empty file.")):
        get_start_lines([])
    with pytest.raises(ValueError, match=re.escape("No JDFTx calculations found in file.")):
        get_start_lines(["\n", "\n"])
    with open(noeigstats_outfile_path, encoding="utf-8") as f:
        outfile_text = list.copy(list(f))
    assert get_start_lines(outfile_text)[0] == 1
    assert len(get_start_lines(outfile_text)) == 1


def test_find_first_range_key():
    """Test the find_first_range_key function.

    This function is used to find the lines that begin with the key, or that start with a pound + spacing
    and then the key.
    """
    out1 = find_first_range_key("barbie", ["barbie"])
    assert len(out1) == 1
    assert out1[0] == 0
    out1 = find_first_range_key("barbie", ["# barbie"])
    assert len(out1) == 0
    out1 = find_first_range_key("barbie", ["# barbie"], skip_pound=True)
    assert len(out1) == 1
    assert out1[0] == 0
    out1 = find_first_range_key("barbie", ["barbie", "barbie", "barbie"])
    assert len(out1) == 3
    out1 = find_first_range_key("barbie", ["barbie", "ken", "barbie"])
    assert len(out1) == 2
    out1 = find_first_range_key("barbie", ["barbie", "ken", "barbie"], startline=1)
    assert len(out1) == 1


def test_get_joutstructures_start_idx():
    """Test the _get_joutstructures_start_idx function.

    This function is used to find the start of the JOutStructures in the outfile.
    This function just matches line with a matched pattern.
    """
    start_flag = "barbie"
    assert _get_joutstructures_start_idx(["ken", "barbie"], out_slice_start_flag=start_flag) == 1
    assert _get_joutstructures_start_idx(["ken", "(unrelated stuff) barbie"], out_slice_start_flag=start_flag) == 1
    assert _get_joutstructures_start_idx(["barbie", "ken"], out_slice_start_flag=start_flag) == 0
    assert _get_joutstructures_start_idx(["ken", "ken"], out_slice_start_flag=start_flag) is None


def test_find_jdftx_out_file(tmp_path):
    """Test the _find_jdftx_out_file function.

    This function is used to find the JDFTx out file in a directory.
    It tests the behavior to make sure the correct errors are raised on directories without and out file
    and directories with multiple out files. And out file must match "*.out" or "out" exactly.
    """
    with pytest.raises(FileNotFoundError, match=re.escape("No JDFTx out file found in directory.")):
        _find_jdftx_out_file(tmp_path)
    write_mt_file(tmp_path, "test.out")
    assert _find_jdftx_out_file(tmp_path) == tmp_path / "test.out"
    # out file has to match "*.out" or "out" exactly
    write_mt_file(tmp_path, "tinyout")
    assert _find_jdftx_out_file(tmp_path) == tmp_path / "test.out"
    os.remove(_find_jdftx_out_file(tmp_path))
    write_mt_file(tmp_path, "out")
    assert _find_jdftx_out_file(tmp_path) == tmp_path / "out"
    write_mt_file(tmp_path, "tinyout.out")
    with pytest.raises(FileNotFoundError, match=re.escape("Multiple JDFTx out files found in directory.")):
        _find_jdftx_out_file(tmp_path)


def test_read_outfile_slices_keeps_last_line(tmp_path):
    """The final line of the out file must be kept in the last slice."""
    lines = ["*************** JDFTx 1\n", "a\n", "*************** JDFTx 2\n", "b\n", "Done!\n"]
    outfile = tmp_path / "jdftx.out"
    outfile.write_text("".join(lines))
    texts = read_outfile_slices(str(outfile))
    assert texts == [lines[:2], lines[2:]]


_NORMALIZED_BANDPROJECTIONS = """2 states, 2 bands, 2 orbital-projections, 1 species
# Symbol nAtoms nOrbitalsPerAtom lMax nShells(l=0) ... nShells(l=lMax)
H 2 1 0 1
#     0  [ +0.0000000 +0.0000000 +0.0000000 ]  0.500000000  spin +1; lines per band, with |projection|^2 per orbital:
0.1 0.2
0.3 0.4
#     1  [ +0.5000000 +0.0000000 +0.0000000 ]  0.500000000  spin +1; lines per band, with |projection|^2 per orbital:
0.5 0.6
0.7 0.8
"""


def test_get_proj_tju_from_file_normalized(tmp_path):
    bandfile = tmp_path / "bandProjections"
    bandfile.write_text(_NORMALIZED_BANDPROJECTIONS)
    proj_tju = get_proj_tju_from_file(bandfile)
    assert proj_tju.dtype == np.float32
    assert_array_equal(proj_tju, np.arange(1, 9, dtype=np.float32).reshape(2, 2, 2) / np.float32(10))
    assert _get_orb_label_list(bandfile) == ("H#1(s)", "H#2(s)")
    assert _get_nbands_from_bandfile_filepath(bandfile) == 2

    # The header is all that is needed for the metadata, so a truncated body does not matter there
    bandfile.write_text("".join(_NORMALIZED_BANDPROJECTIONS.splitlines(keepends=True)[:5]))
    assert _get_orb_label_list(bandfile) == ("H#1(s)", "H#2(s)")
    with pytest.raises(RuntimeError, match="does not match expected length"):
        get_proj_tju_from_file(bandfile)

    bandfile.write_text("")
    with pytest.raises(ValueError, match="Provided an empty file"):
        _get_nbands_from_bandfile_filepath(bandfile)


def test_get_proj_tju_from_file_complex():
    bandfile = n2_ex_calc_dir / "bandProjections"
    proj_tju = get_proj_tju_from_file(bandfile)
    assert proj_tju.dtype == np.complex64

    lines = bandfile.read_text().splitlines()
    nstates, nbands, nproj = (int(lines[0].split()[i]) for i in (0, 2, 4))
    assert proj_tju.shape == (nstates, nbands, nproj)
    for istate, iband in [(0, 0), (3, 7), (nstates - 1, nbands - 1)]:
        values = np.array(lines[3 + istate * (nbands + 1) + 1 + iband].split(), dtype=np.float32)
        assert_array_equal(proj_tju[istate, iband].real, values[::2])
        assert_array_equal(proj_tju[istate, iband].imag, values[1::2])
