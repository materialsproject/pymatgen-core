from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pytest

from pymatgen.core import Molecule
from pymatgen.core.structure import Structure
from pymatgen.io.xcrysden import XSF, XSFBand, XSFGrid
from pymatgen.util.testing import TEST_FILES_DIR, MatSciTest


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
            data=np.ones((1, 2, 2, 2)),
            lattice=np.eye(3),
            origin=np.zeros(3),
            comment="fermi surface",
            labels=["grid/1"],
        )
        band_roundtrip = XSFBand.from_dict(band.as_dict())
        assert band_roundtrip.comment == "fermi surface"
        assert band_roundtrip.labels == ["grid/1"]
        np.testing.assert_allclose(band_roundtrip.data, band.data)

        with pytest.raises(ValueError, match="XSFGrid labels must be empty or match"):
            XSFGrid(data=np.ones((2, 2, 3)), lattice=np.eye(2, 3), origin=np.zeros(3), labels=["only one label"])

        with pytest.raises(ValueError, match="labels must be empty or match"):
            XSFBand(
                data=np.ones((2, 2, 2, 2)),
                lattice=np.eye(3),
                origin=np.zeros(3),
                labels=["only one label"],
            )

    def test_to_str_roundtrip_with_forces_grid_and_band(self):
        xsf = XSF(self.struct.copy())
        xsf.forces = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        xsf.grids["density"] = XSFGrid(
            data=np.arange(6, dtype=float).reshape(1, 2, 3),
            lattice=np.eye(2, 3),
            origin=np.zeros(3),
            labels=["grid/rho"],
        )
        xsf.fermi_energy = 5.0
        xsf.bands["bands"] = XSFBand(
            data=np.arange(8, dtype=float).reshape(1, 2, 2, 2),
            lattice=np.eye(3),
            origin=np.zeros(3),
            labels=["band_1"],
        )

        roundtrip = XSF.from_str(xsf.to_str())
        np.testing.assert_allclose(roundtrip.forces, xsf.forces)
        np.testing.assert_allclose(roundtrip.grids["density"].data, xsf.grids["density"].data)
        np.testing.assert_allclose(roundtrip.bands["bands"].data, xsf.bands["bands"].data)
        assert roundtrip.fermi_energy == 5.0

    def test_from_file_reads_fixture(self):
        xsf = XSF.from_file(Path(TEST_FILES_DIR) / "io" / "xcrysden" / "crystal_primvec_primcoord.xsf")

        assert xsf.kind == "crystal"
        assert xsf.structure is not None
        assert len(xsf.structure) == 2

    def test_parse_file_supports_file_streams(self):
        fixture = Path(TEST_FILES_DIR) / "io" / "xcrysden" / "crystal_primvec_primcoord.xsf"
        with open(fixture, "rb") as file:
            xsf_binary = XSF.parse_file(file)
        with open(fixture, encoding="utf-8") as file:
            xsf_text = XSF.parse_file(file)

        assert xsf_binary.kind == "crystal"
        assert xsf_text.kind == "crystal"
        assert xsf_binary.structure == xsf_text.structure

    def test_parse_file_rejects_string_io(self):
        with pytest.raises(TypeError, match="binary stream"):
            XSF.parse_file(StringIO("CRYSTAL\n"))

    def test_datagrid_roundtrip_preserves_values(self):
        with open(Path(TEST_FILES_DIR) / "io" / "xcrysden" / "datagrid_2d.xsf", "rb") as file:
            xsf_2d = XSF.parse_file(file)
        with open(Path(TEST_FILES_DIR) / "io" / "xcrysden" / "datagrid_3d.xsf", "rb") as file:
            xsf_3d = XSF.parse_file(file)

        assert xsf_2d.grids["density_2d"].data.tolist() == [[[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]]]
        assert xsf_3d.grids["density_3d"].data.tolist() == [[[[0.0, 4.0], [2.0, 6.0]], [[1.0, 5.0], [3.0, 7.0]]]]

    def test_molecule_atoms_roundtrip(self):
        molecule = Molecule(["O", "H", "H"], [[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]])
        xsf = XSF(molecule, forces=np.ones((3, 3)))

        roundtrip = XSF.from_str(xsf.to_str())

        assert roundtrip.kind == "molecule"
        assert roundtrip.lattice is None
        assert isinstance(roundtrip.structure, Molecule)
        assert roundtrip.structure.composition == molecule.composition
        np.testing.assert_allclose(roundtrip.structure.cart_coords, molecule.cart_coords)
        np.testing.assert_allclose(roundtrip.forces, np.ones((3, 3)))

    def test_structure_from_str_rejects_molecule_xsf(self):
        with pytest.raises(ValueError, match="XSF data contains a Molecule"):
            Structure.from_str(
                """MOLECULE
ATOMS
O 0.0 0.0 0.0
H 0.0 0.0 1.0
""",
                fmt="xsf",
            )

    def test_structure_from_str_stores_xsf_extras_in_properties(self):
        xsf = XSF(self.struct.copy())
        xsf.grids["density"] = XSFGrid(
            data=np.arange(6, dtype=float).reshape(1, 2, 3),
            lattice=np.eye(2, 3),
            origin=np.zeros(3),
            labels=["grid/rho"],
        )
        xsf.fermi_energy = 5.0
        xsf.bands["bands"] = XSFBand(
            data=np.arange(8, dtype=float).reshape(1, 2, 2, 2),
            lattice=np.eye(3),
            origin=np.zeros(3),
            labels=["band_1"],
        )
        struct = Structure.from_str(xsf.to_str(), fmt="xsf")

        assert struct.composition == self.struct.composition
        np.testing.assert_allclose(struct.lattice.matrix, self.struct.lattice.matrix)
        np.testing.assert_allclose(struct.frac_coords, self.struct.frac_coords)
        assert isinstance(struct.properties["grids/density/rho"], XSFGrid)
        assert isinstance(struct.properties["bands/bands/band_1"], XSFBand)
        assert struct.properties["bands/fermi_energy"] == 5.0
        np.testing.assert_allclose(struct.properties["grids/density/rho"].data, xsf.grids["density"].data)
        np.testing.assert_allclose(struct.properties["bands/bands/band_1"].data, xsf.bands["bands"].data)

    def test_atoms_stops_before_following_section(self):
        xsf = XSF.from_str(
            """MOLECULE
ATOMS
O 0.0 0.0 0.0
H 0.0 0.0 1.0
BEGIN_BLOCK_DATAGRID_3D
density
BEGIN_DATAGRID_3D_rho
1 1 1
0 0 0
1 0 0
0 1 0
0 0 1
0.0
END_DATAGRID_3D
END_BLOCK_DATAGRID_3D
"""
        )

        assert isinstance(xsf.structure, Molecule)
        assert len(xsf.structure) == 2
        assert "density" in xsf.grids

    def test_xsf_rejects_atoms_before_kind(self):
        with pytest.raises(ValueError, match="ATOMS is only valid in MOLECULE sections"):
            XSF.from_str(
                """ATOMS
1 1
H 0.0 0.0 0.0
"""
            )

    def test_xsf_rejects_primcoord_before_primvec(self):
        with pytest.raises(ValueError, match="PRIMCOORD encountered before PRIMVEC"):
            XSF.from_str(
                """CRYSTAL
PRIMCOORD
1 1
H 0.0 0.0 0.0
"""
            )

    def test_xsf_rejects_convvect(self):
        with pytest.raises(NotImplementedError, match="CONVCOORD section is not allowed in XSF files"):
            XSF.from_str(
                """CRYSTAL
PRIMVEC
 1 0 0
 0 1 0
 0 0 1
CONVCOORD
 1 0 0
 0 1 0
 0 0 1
PRIMCOORD
1 1
H 0 0 0
"""
            )

    def test_xsf_rejects_malformed_datagrid(self):
        with pytest.raises(ValueError, match=r"Unsupported DATAGRID dimensionality|No data parsed"):
            XSF.from_str(
                """CRYSTAL
PRIMVEC
 1 0 0
 0 1 0
 0 0 1
PRIMCOORD
1 1
H 0 0 0
BEGIN_BLOCK_DATAGRID_4D
block
END_BLOCK_DATAGRID_4D
"""
            )

    def test_xsf_rejects_bandgrid_without_fermi_energy(self):
        with pytest.raises(ValueError, match="BANDGRID block is missing required Fermi energy"):
            XSF.from_str(
                """BEGIN_BLOCK_BANDGRID_3D
band_energies
BEGIN_BANDGRID_3D_band
1
1 1 1
0 0 0
1 0 0
0 1 0
0 0 1
BAND: 1
0.0
END_BANDGRID_3D
END_BLOCK_BANDGRID_3D
"""
            )

    def test_xsf_rejects_end_info_without_begin_info(self):
        with pytest.raises(ValueError, match="END_INFO encountered without a preceding BEGIN_INFO"):
            XSF.from_str(
                """END_INFO
"""
            )

    def test_xsf_rejects_multiple_begin_info_sections(self):
        with pytest.raises(ValueError, match="Multiple BEGIN_INFO sections are not supported"):
            XSF.from_str(
                """BEGIN_INFO
Fermi Energy: 1.0
END_INFO
BEGIN_INFO
Fermi Energy: 2.0
END_INFO
"""
            )

    def test_xsf_rejects_end_block_datagrid_without_begin(self):
        with pytest.raises(ValueError, match="END_BLOCK_DATAGRID encountered without a matching BEGIN_BLOCK_DATAGRID"):
            XSF.from_str(
                """END_BLOCK_DATAGRID_3D
"""
            )

    def test_xsf_rejects_end_block_bandgrid_without_begin(self):
        with pytest.raises(ValueError, match="END_BLOCK_BANDGRID encountered without a matching BEGIN_BLOCK_BANDGRID"):
            XSF.from_str(
                """END_BLOCK_BANDGRID_3D
"""
            )

    def test_xsf_rejects_begin_datagrid_without_block(self):
        with pytest.raises(ValueError, match="BEGIN_DATAGRID encountered without a matching BEGIN_BLOCK_DATAGRID"):
            XSF.from_str(
                """BEGIN_DATAGRID_3D_density
1 1 1
0 0 0
1 0 0
0 1 0
0 0 1
0.0
"""
            )

    def test_xsf_rejects_begin_bandgrid_without_block(self):
        with pytest.raises(ValueError, match="BEGIN_BANDGRID encountered without a matching BEGIN_BLOCK_BANDGRID"):
            XSF.from_str(
                """BEGIN_INFO
Fermi Energy: 1.0
END_INFO
BEGIN_BANDGRID_3D_band
1
1 1 1
0 0 0
1 0 0
0 1 0
0 0 1
BAND: 1
0.0
END_BANDGRID_3D
"""
            )

    def test_xsf_rejects_bad_primcoord_field_count(self):
        with pytest.raises(ValueError, match="PRIMCOORD atom rows must contain 4 fields or 7 fields with forces"):
            XSF.from_str(
                """CRYSTAL
PRIMVEC
 1 0 0
 0 1 0
 0 0 1
PRIMCOORD
1 1
H 0 0
"""
            )

    def test_xsf_rejects_truncated_datagrid(self):
        with pytest.raises(ValueError, match="Expected 8 grid values but parsed 7"):
            XSF.from_str(
                """CRYSTAL
PRIMVEC
 1 0 0
 0 1 0
 0 0 1
PRIMCOORD
1 1
H 0 0 0
BEGIN_BLOCK_DATAGRID_3D
rho
BEGIN_DATAGRID_3D_rho
2 2 2
0 0 0
1 0 0
0 1 0
0 0 1
0 1 2 3 4 5 6
END_DATAGRID_3D
END_BLOCK_DATAGRID_3D
"""
            )

    def test_xsf_rejects_nested_datagrid_block(self):
        with pytest.raises(ValueError, match="Nested BEGIN_BLOCK_DATAGRID is not allowed"):
            XSF.from_str(
                """BEGIN_BLOCK_DATAGRID_3D
block1
BEGIN_BLOCK_DATAGRID_3D
block2
"""
            )

    def test_xsf_rejects_nested_bandgrid_block(self):
        with pytest.raises(ValueError, match="Nested BEGIN_BLOCK_BANDGRID is not allowed"):
            XSF.from_str(
                """BEGIN_INFO
Fermi Energy: 1.0
END_INFO
BEGIN_BLOCK_BANDGRID_3D
bands
BEGIN_BLOCK_BANDGRID_3D
bands2
"""
            )

    def test_xsf_rejects_bandgrid_band_count_mismatch(self):
        with pytest.raises(ValueError, match="Expected 2 bands but parsed 1"):
            XSF.from_str(
                """BEGIN_INFO
  Fermi Energy: 1.0
END_INFO
BEGIN_BLOCK_BANDGRID_3D
bands
BEGIN_BANDGRID_3D_band
2
1 1 1
0 0 0
1 0 0
0 1 0
0 0 1
BAND: 1
0.0
END_BANDGRID_3D
END_BLOCK_BANDGRID_3D
"""
            )

    def test_xsf_rejects_duplicate_datagrid_block_name(self):
        with pytest.raises(ValueError, match="Duplicate DATAGRID block name: density"):
            XSF.from_str(
                """CRYSTAL
PRIMVEC
 1 0 0
 0 1 0
 0 0 1
PRIMCOORD
1 1
H 0 0 0
BEGIN_BLOCK_DATAGRID_3D
density
BEGIN_DATAGRID_3D_rho
1 1 1
0 0 0
1 0 0
0 1 0
0 0 1
0.0
END_DATAGRID_3D
END_BLOCK_DATAGRID_3D
BEGIN_BLOCK_DATAGRID_3D
density
BEGIN_DATAGRID_3D_rho2
1 1 1
0 0 0
1 0 0
0 1 0
0 0 1
1.0
END_DATAGRID_3D
END_BLOCK_DATAGRID_3D
"""
            )
