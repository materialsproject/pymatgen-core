from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pymatgen.core import Lattice, Structure
from pymatgen.io.common import PMGDir, VolumetricData
from pymatgen.util.testing import TEST_FILES_DIR

if TYPE_CHECKING:
    from pathlib import Path


def test_cube_io_faithful(tmp_path: Path) -> None:
    in_path = f"{TEST_FILES_DIR}/io/cube-gh-2817.xyz"

    cube_file = VolumetricData.from_cube(in_path)
    out_path = f"{tmp_path}/cube-gh-2817.xyz"
    cube_file.to_cube(out_path)
    out_cube = VolumetricData.from_cube(out_path)

    # structure should be preserved round-trip to/from cube file
    assert cube_file.structure.volume == out_cube.structure.volume
    assert cube_file.structure == out_cube.structure


def test_cube_nonzero_origin() -> None:
    """Regression test for gh-4607: from_cube must account for the grid origin on line 3."""
    vd = VolumetricData.from_cube(f"{TEST_FILES_DIR}/electronic_structure/boltztrap/fermi/boltztrap_BZ.cube")

    assert len(vd.structure) == 96
    assert str(vd.structure[0].specie) == "Fe"
    np.testing.assert_allclose(
        vd.structure[0].frac_coords,
        [0.16528953, 0.6611559, 0.24615391],
        atol=1e-6,
    )


def test_volumetric_data_periodic_interpolation() -> None:
    """Regression test for gh-3787: volumetric grids use periodic i/n sampling."""
    shape = (2, 3, 4)
    grid_indices = np.indices(shape)
    data = grid_indices[0] + 10 * grid_indices[1] + 100 * grid_indices[2]
    volumetric_data = VolumetricData(
        Structure(Lattice.cubic(1), [], []),
        {"total": data},
    )

    np.testing.assert_allclose(volumetric_data.xpoints, [0, 0.5])
    np.testing.assert_allclose(volumetric_data.ypoints, [0, 1 / 3, 2 / 3])
    np.testing.assert_allclose(volumetric_data.zpoints, [0, 0.25, 0.5, 0.75])

    # Values at grid points should be returned exactly.
    assert volumetric_data.value_at(0.5, 1 / 3, 0.5) == pytest.approx(211)

    # Interpolation across every periodic boundary should use the samples at index 0.
    boundary_point = np.array([0.75, 5 / 6, 0.875])
    assert volumetric_data.value_at(*boundary_point) == pytest.approx(160.5)
    assert volumetric_data.value_at(*(boundary_point - 1)) == pytest.approx(160.5)
    assert volumetric_data.value_at(1, 1, 1) == pytest.approx(data[0, 0, 0])
    assert volumetric_data.value_at(-1e-18, 0, 0) == pytest.approx(data[0, 0, 0])

    # Integer lattice translations must not change interpolated values.
    translations = np.array([[0, 0, 0], [1, -2, 3], [-4, 2, -1]])
    translated_values = volumetric_data.interpolator(boundary_point + translations)
    np.testing.assert_allclose(translated_values, 160.5)

    slice_end = boundary_point + np.array([1, 0, 0])
    slice_values = volumetric_data.linear_slice(boundary_point, slice_end, n=5)
    assert isinstance(slice_values, list)
    np.testing.assert_allclose(slice_values, volumetric_data.interpolator(np.linspace(boundary_point, slice_end, 5)))

    # Interpolation should use the current grid rather than a stale cached copy.
    volumetric_data.scale(2)
    assert volumetric_data.value_at(*boundary_point) == pytest.approx(321)

    with pytest.raises(ValueError, match=r"shape \(\.\.\., 3\)"):
        volumetric_data.interpolator([0, 0])


def test_volumetric_data_interpolation_with_singleton_dimensions() -> None:
    volumetric_data = VolumetricData(
        Structure(Lattice.cubic(1), [], []),
        {"total": np.array([[[2]], [[6]]])},
    )

    assert volumetric_data.value_at(0.75, 0.4, -0.2) == pytest.approx(4)


class TestPMGDir:
    def test_getitem(self):
        # Some simple testing of loading and reading since all these were tested in other classes.
        d = PMGDir(f"{TEST_FILES_DIR}/io/vasp/fixtures/relaxation")
        assert len(d) == 5
        assert d["OUTCAR"].run_stats["cores"] == 8

        d = PMGDir(f"{TEST_FILES_DIR}/io/vasp/fixtures/scan_relaxation")
        assert len(d) == 2
        assert "vasprun.xml.gz" in d
        assert "OUTCAR" in d
        assert d["vasprun.xml.gz"].incar["METAGGA"] == "R2scan"

        with pytest.raises(ValueError, match="hello not found"):
            d["hello"]

        d = PMGDir(f"{TEST_FILES_DIR}/io/pwscf")
        with pytest.warns(UserWarning, match=r"No parser defined for Si.pwscf.out"):
            assert isinstance(d["Si.pwscf.out"], str)

        # Test NEB directories.
        d = PMGDir(f"{TEST_FILES_DIR}/io/vasp/fixtures/neb_analysis/neb1/neb")

        assert len(d) == 10
        from pymatgen.io.vasp import Poscar

        assert isinstance(d["00/POSCAR"], Poscar)

        outcars = d.get_files_by_name("OUTCAR")
        assert len(outcars) == 5
        assert all("OUTCAR" for k in outcars)

        d.reset()
        for v in d._files.values():
            assert v is None
