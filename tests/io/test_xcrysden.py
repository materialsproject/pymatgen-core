from __future__ import annotations

import numpy as np
import pytest

from pymatgen.core.structure import Structure
from pymatgen.io.xcrysden import XSF, XSFBand, XSFGrid
from pymatgen.util.testing import MatSciTest


class TestXSF(MatSciTest):
    def setup_method(self):
        self.coords = [[0, 0, 0], [0.75, 0.5, 0.75]]
        self.lattice = [
            [3.8401979337, 0.00, 0.00],
            [1.9200989668, 3.3257101909, 0.00],
            [0.00, -2.2171384943, 3.1355090603],
        ]
        self.struct = Structure(self.lattice, ["Si", "Si"], self.coords)

    def test_xsf(self):
        xsf = XSF(self.struct)
        assert self.struct, XSF.from_str(xsf.to_str())
        xsf = XSF(self.struct)
        assert self.struct, XSF.from_str(xsf.to_str())

    def test_append_vect(self):
        self.struct.add_site_property("vect", np.eye(2, 3))
        xsf_str = XSF(self.struct).to_str()
        last_line_split = xsf_str.split("\n")[-1].split()
        assert len(last_line_split) == 7
        assert last_line_split[-1] == "0.00000000000000"
        assert last_line_split[-2] == "1.00000000000000"
        assert last_line_split[-3] == "0.00000000000000"

    def test_to_str(self):
        structure = self.get_structure("Li2O")
        xsf = XSF(structure)
        assert (
            xsf.to_str()
            == """CRYSTAL
# Primitive lattice vectors in Angstrom
PRIMVEC
 2.91738857000000 0.09789437000000 1.52000466000000
 0.96463406000000 2.75503561000000 1.52000466000000
 0.13320635000000 0.09789443000000 3.28691771000000
# Cartesian coordinates in Angstrom.
PRIMCOORD
 3 1
O     0.00000000000000     0.00000000000000     0.00000000000000
Li     3.01213761017484     2.21364440998406     4.74632330032018
Li     1.00309136982516     0.73718000001594     1.58060372967982"""
        )

        assert (
            xsf.to_str(atom_symbol=False)
            == """CRYSTAL
# Primitive lattice vectors in Angstrom
PRIMVEC
 2.91738857000000 0.09789437000000 1.52000466000000
 0.96463406000000 2.75503561000000 1.52000466000000
 0.13320635000000 0.09789443000000 3.28691771000000
# Cartesian coordinates in Angstrom.
PRIMCOORD
 3 1
8     0.00000000000000     0.00000000000000     0.00000000000000
3     3.01213761017484     2.21364440998406     4.74632330032018
3     1.00309136982516     0.73718000001594     1.58060372967982"""
        )

    def test_xsf_symbol_parse(self):
        """Ensure that the same structure is parsed
        even if the atomic symbol / number convention
        is different.
        """
        test_str = """
CRYSTAL
PRIMVEC
       11.45191956     0.00000000     0.00000000
        5.72596044     9.91765288     0.00000000
      -14.31490370    -8.26471287    23.37613199
PRIMCOORD
1 1
H     -0.71644986    -0.41364333     1.19898200     0.00181803     0.00084718     0.00804832
"""
        structure = XSF.from_str(test_str).structure
        assert str(structure.species[0]) == "H"
        test_string2 = """
CRYSTAL
PRIMVEC
       11.45191956     0.00000000     0.00000000
        5.72596044     9.91765288     0.00000000
      -14.31490370    -8.26471287    23.37613199
PRIMCOORD
1 1
1     -0.71644986    -0.41364333     1.19898200     0.00181803     0.00084718     0.00804832
"""

        structure2 = XSF.from_str(test_string2).structure
        assert structure == structure2

    def test_structure_from_str_rejects_xsf_without_structure(self, monkeypatch):
        xsf = XSF()
        xsf.grids["block_name"] = XSFGrid(
            data=np.zeros((1, 1, 1)),
            lattice=np.eye(3),
            origin=np.zeros(3),
        )
        monkeypatch.setattr(XSF, "from_str", lambda *args, **kwargs: xsf)

        with pytest.raises(ValueError, match="XSF data does not contain a structure"):
            Structure.from_str("grid only", fmt="xsf")

    def test_grid_and_band_are_msonable(self):
        grid = XSFGrid(
            data=np.ones((1, 2, 3)),
            lattice=np.eye(2, 3),
            origin=np.zeros(3),
            comment="rho",
            labels=["grid/rho"],
        )
        grid_roundtrip = XSFGrid.from_dict(grid.as_dict())
        assert grid_roundtrip.comment == "rho"
        assert grid_roundtrip.labels == ["grid/rho"]
        assert grid_roundtrip.ndim == 2
        np.testing.assert_allclose(grid_roundtrip.data, grid.data)
        np.testing.assert_allclose(grid_roundtrip.lattice, grid.lattice)
        np.testing.assert_allclose(grid_roundtrip.origin, grid.origin)

        band = XSFBand(
            fermi_energy=1.5,
            data=np.ones((1, 2, 2, 2)),
            lattice=np.eye(3),
            origin=np.zeros(3),
            comment="fermi surface",
            labels=["grid/1"],
        )
        band_roundtrip = XSFBand.from_dict(band.as_dict())
        assert band_roundtrip.fermi_energy == 1.5
        assert band_roundtrip.comment == "fermi surface"
        assert band_roundtrip.labels == ["grid/1"]
        np.testing.assert_allclose(band_roundtrip.data, band.data)

        with pytest.raises(ValueError, match="XSFGrid labels must be empty or match"):
            XSFGrid(data=np.ones((2, 2, 3)), lattice=np.eye(2, 3), origin=np.zeros(3), labels=["only one label"])

        with pytest.raises(ValueError, match="labels must be empty or match"):
            XSFBand(
                fermi_energy=1.5,
                data=np.ones((2, 2, 2, 2)),
                lattice=np.eye(3),
                origin=np.zeros(3),
                labels=["only one label"],
            )
